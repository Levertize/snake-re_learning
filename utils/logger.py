# =============================================================================
# logger.py — Logging ke file dan terminal
#
# Mencatat progress training ke dua tempat:
#   1. Terminal (stdout) — untuk monitoring real-time
#   2. File log — untuk analisis post-training
#
# Format log:
#   [Episode XXX] Score: X | Record: X | Mean: X.XX | Epsilon: X.XXXX | Memory: XXXXX
# =============================================================================

import os
import logging
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


def setup_logger(name='snake_rl'):
    """
    Setup logger yang menulis ke terminal DAN file secara bersamaan.

    File log disimpan di folder logs/ dengan nama berisi timestamp,
    sehingga setiap sesi training punya file log terpisah.

    Args:
        name (str): Nama logger

    Returns:
        logging.Logger: Logger yang sudah dikonfigurasi
    """
    logger = logging.getLogger(name)

    # Hindari duplikasi handler jika setup_logger dipanggil lebih dari sekali
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Buat folder logs/ jika belum ada
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # File handler — simpan log ke file dengan timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config.LOG_DIR, f'training_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Console handler — tampilkan di terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format log
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"=== Snake RL Training Log ===")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'=' * 40}")

    return logger


def log_episode(logger, episode, score, record, mean_score, epsilon, memory_size):
    """
    Log informasi satu episode training.

    Args:
        logger (logging.Logger): Logger instance
        episode (int): Nomor episode
        score (int): Score episode ini
        record (int): Score rekor tertinggi
        mean_score (float): Rata-rata score sampai saat ini
        epsilon (float): Nilai epsilon saat ini
        memory_size (int): Jumlah experience di replay buffer
    """
    logger.info(
        f"[Episode {episode:>4d}] "
        f"Score: {score:>3d} | "
        f"Record: {record:>3d} | "
        f"Mean: {mean_score:>6.2f} | "
        f"Epsilon: {epsilon:.4f} | "
        f"Memory: {memory_size:>6d}"
    )
