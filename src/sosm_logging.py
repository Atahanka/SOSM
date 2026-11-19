import logging
import os
from datetime import datetime

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Initializes dual file and console loggers with timestamps."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sosm_{timestamp}.log")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("SOSM")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_training_start(logger, model, loader, epochs, device):
    logger.info("=" * 60)
    logger.info("SOSM Training Started")
    logger.info(f"Device: {device} | Batch size: {loader.batch_size} | Epochs: {epochs}")
    logger.info(f"Model: {model.__class__.__name__} | Latent dim: {model.encoder.rna_enc.proj[0].out_features}")
    logger.info(f"λ_curv: {model.lambda_curv}, Temp: {model.temp}, Sigma: {model.sigma}")

def log_epoch(logger, epoch, loss, align, curv, lr):
    logger.info(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Align: {align:.4f} | Curv: {curv:.4f} | LR: {lr:.6f}")

def log_training_end(logger, losses):
    logger.info("=" * 60)
    logger.info(f"Training Complete. Final loss: {losses[-1]:.4f}")
    logger.info(f"Loss decrease: {losses[0] - losses[-1]:.4f}")