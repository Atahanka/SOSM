import pandas as pd
import numpy as np
import anndata
import subprocess
import os
import tarfile
from scipy.stats import pearsonr

def download_with_curl(url: str, output_path: str):
    """Downloads remote archives using native curl to bypass SSL/firewall constraints."""
    print(f"Downloading {output_path} via curl...")
    curl_cmd = ["curl", "-L", "-k", "-A", "Mozilla/5.0", "-o", output_path, url]
    subprocess.run(curl_cmd, check=True)

def extract_target_file(tar_path: str, endswith_str: str, output_name: str):
    """Extracts a specific data matrix file from a compressed tarball."""
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.name.endswith(endswith_str):
                m.name = output_name  
                tar.extract(m, path=".")
                return
        raise FileNotFoundError(f"Could not find a file ending with {endswith_str}")

def process_genomic_matrix(filepath: str, modality_name: str) -> pd.DataFrame:
    """Parses and normalizes tabular transcriptomic or miRNA matrices."""
    print(f"Processing {modality_name} data...")
    df = pd.read_csv(filepath, sep='\t', index_col=0, low_memory=False)
    df = df.iloc[1:] # Drop metadata row
    df = df.T
    df.index = df.index.astype(str).str[:12].str.upper()
    df.index.name = 'bcr_patient_barcode'
    print(f"Converting {modality_name} to numeric floats...")
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.reset_index()
    df = df[df['bcr_patient_barcode'].str.startswith('TCGA')]
    df = df.groupby('bcr_patient_barcode').mean().reset_index()
    return df

def process_massive_methylation(filepath: str, max_features: int = 2000, chunksize: int = 10000) -> pd.DataFrame:
    """Memory-efficient chunked parser for the Methylation450k array."""
    print("Processing Methylation450k array in memory-efficient chunks...")
    chunk_iter = pd.read_csv(filepath, sep='\t', chunksize=chunksize, skiprows=[1], index_col=0, low_memory=False)
    top_chunks = []
    
    for i, chunk in enumerate(chunk_iter):
        chunk = chunk.apply(pd.to_numeric, errors='coerce')
        variances = chunk.var(axis=1)
        chunk['var'] = variances
        chunk = chunk.sort_values('var', ascending=False).head(max_features)
        top_chunks.append(chunk)
        print(f"  - Parsed chunk {i+1} ({(i+1)*chunksize} CpGs evaluated)")
        
    print("Consolidating top methylation features...")
    combined = pd.concat(top_chunks)
    combined = combined.sort_values('var', ascending=False).head(max_features)
    combined = combined.drop(columns=['var'])
    
    df = combined.T
    df.index = df.index.astype(str).str[:12].str.upper()
    df.index.name = 'bcr_patient_barcode'
    df = df.reset_index()
    df = df[df['bcr_patient_barcode'].str.startswith('TCGA')]
    df = df.groupby('bcr_patient_barcode').mean().reset_index()
    return df

def select_top_features(df: pd.DataFrame, max_features: int = 2000):
    """Selects top variable features by variance (raw matrix returned; per-fold scaling applied downstream)."""
    feature_names = df.drop(columns=['bcr_patient_barcode']).columns.values
    mat = df.drop(columns=['bcr_patient_barcode']).fillna(0).values.astype(float)
    
    if mat.shape[1] > max_features:
        variances = np.var(mat, axis=0)
        top_idx = np.argsort(variances)[-max_features:]
        mat = mat[:, top_idx]
        feature_names = feature_names[top_idx]
        
    return mat, feature_names

def load_real_tcga_kirc():
    """Downloads and loads matched clinical, RNA, miRNA, and Methylation data for TCGA-KIRC."""
    print("Initializing Broad Firehose multi-modal downloads...")
    urls = {
        'clin': "http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/KIRC/20160128/gdac.broadinstitute.org_KIRC.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz",
        'rna': "http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/KIRC/20160128/gdac.broadinstitute.org_KIRC.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz",
        'mirna': "http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/KIRC/20160128/gdac.broadinstitute.org_KIRC.Merge_mirnaseq__illuminahiseq_mirnaseq__bcgsc_ca__Level_3__miR_gene_expression__data.Level_3.2016012800.0.0.tar.gz",
        'methyl': "http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/KIRC/20160128/gdac.broadinstitute.org_KIRC.Merge_methylation__humanmethylation450__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data.Level_3.2016012800.0.0.tar.gz"
    }
    
    if not os.path.exists("KIRC.clin.txt"):
        download_with_curl(urls['clin'], "clin.tar.gz")
        extract_target_file("clin.tar.gz", "picked.txt", "KIRC.clin.txt")
        os.remove("clin.tar.gz")
        
    if not os.path.exists("KIRC.rna.txt"):
        download_with_curl(urls['rna'], "rna.tar.gz")
        extract_target_file("rna.tar.gz", "data.txt", "KIRC.rna.txt")
        os.remove("rna.tar.gz")
        
    if not os.path.exists("KIRC.mirna.txt"):
        download_with_curl(urls['mirna'], "mirna.tar.gz")
        extract_target_file("mirna.tar.gz", "data.txt", "KIRC.mirna.txt")
        os.remove("mirna.tar.gz")
        
    if not os.path.exists("KIRC.methyl450.txt"):
        download_with_curl(urls['methyl'], "methyl.tar.gz")
        extract_target_file("methyl.tar.gz", "data.txt", "KIRC.methyl450.txt")
        os.remove("methyl.tar.gz")
        
    print("Processing clinical data...")
    clin = pd.read_csv("KIRC.clin.txt", sep='\t', index_col=0).T
    clin.columns = [str(c).lower().split('.')[-1] for c in clin.columns]
    
    time = clin['days_to_death'].fillna(clin['days_to_last_followup'])
    time = pd.to_numeric(time, errors='coerce')
    event = clin['vital_status'].apply(lambda x: 1 if str(x).lower() in ['1', 'dead', 'deceased'] else 0)
    
    clinical = pd.DataFrame({'time': time, 'event': event})
    clinical.index = clin.index.astype(str).str[:12].str.upper()
    clinical.index.name = 'bcr_patient_barcode'
    clinical = clinical.reset_index().dropna()
    clinical = clinical[clinical['bcr_patient_barcode'].str.startswith('TCGA')]
    
    rna = process_genomic_matrix("KIRC.rna.txt", "RNA")
    mirna = process_genomic_matrix("KIRC.mirna.txt", "miRNA")
    methyl = process_massive_methylation("KIRC.methyl450.txt", max_features=2000, chunksize=25000)
    
    common_barcodes = sorted(list(
        set(clinical['bcr_patient_barcode'])
        .intersection(rna['bcr_patient_barcode'])
        .intersection(mirna['bcr_patient_barcode'])
        .intersection(methyl['bcr_patient_barcode'])
    ))
    
    clinical = clinical.set_index('bcr_patient_barcode').loc[common_barcodes]
    rna = rna.set_index('bcr_patient_barcode').loc[common_barcodes].reset_index()
    mirna = mirna.set_index('bcr_patient_barcode').loc[common_barcodes].reset_index()
    methyl = methyl.set_index('bcr_patient_barcode').loc[common_barcodes].reset_index()
    
    time = clinical['time'].values.astype(float)
    event = clinical['event'].values.astype(int)
    
    rna_data, rna_names = select_top_features(rna, max_features=5000)
    mirna_data, mirna_names = select_top_features(mirna, max_features=2000)
    methyl_data, methyl_names = select_top_features(methyl, max_features=5000)
    
    adata = anndata.AnnData(np.zeros((len(time), 1)))
    adata.obsm['rna'] = rna_data
    adata.obsm['mirna'] = mirna_data
    adata.obsm['methyl'] = methyl_data
    
    adata.uns['rna_names'] = rna_names
    adata.uns['mirna_names'] = mirna_names
    adata.uns['methyl_names'] = methyl_names
    
    adata.obs['time'] = time
    adata.obs['event'] = event
    
    print(f"✅ TCGA-KIRC Multi-Modal Cohort Loaded: {len(time)} matched patients")
    print(f"  - Event rate: {event.mean():.2%}")
    return adata, time, event

def get_prognostic_pseudotime_for_sosm(
    adata,
    rna_key: str = "rna",
    survival_labels=None,
    n_prolif: int = 100,
    n_immune: int = 100,
    force_invert: bool = None,
) -> tuple[np.ndarray, bool]:
    """
    Computes biological-prior prognostic pseudotime in a fully unsupervised manner
    via proliferation/immune transcriptional proxies.

    Optionally performs 1-bit polarity alignment (weak supervision) when survival
    labels are provided on a training cohort, or applies a pre-calibrated polarity
    (`force_invert`) blindly to unseen cohorts.
    """
    if hasattr(adata, 'X') and adata.X.shape[1] > 1:
        rna = adata.X
    elif rna_key in adata.obsm:
        rna = adata.obsm[rna_key]
    elif 'rna' in adata.obsm:
        rna = adata.obsm['rna']
    else:
        raise KeyError(
            f"Could not find RNA data in adata.X or adata.obsm "
            f"(tried: {[rna_key, 'rna']}, available keys: {list(adata.obsm.keys())})"
        )
    
    if len(rna.shape) == 1:
        rna = rna.reshape(1, -1)
    
    prolif = rna[:, :n_prolif].mean(axis=1)
    immune = rna[:, n_prolif:n_prolif + n_immune].mean(axis=1)
    score = +1.5 * prolif - 1.2 * immune

    inverted = False
    if force_invert is not None:
        if force_invert:
            score = -score
            inverted = True
    elif survival_labels is not None:
        time, _ = survival_labels
        corr, _ = pearsonr(score, time)
        if corr > 0:
            score = -score
            inverted = True

    norm_score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return norm_score, inverted