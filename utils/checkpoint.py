# =============================================================================
# checkpoint.py — Save & Load model checkpoint
#
# Auto-save model weights setiap kali agent mencetak rekor baru.
# Mendukung resume training dari checkpoint yang sudah disimpan.
#
# Format file: .pth (PyTorch state dict + metadata)
# Lokasi default: models/best_model.pth
# =============================================================================

import os
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


def save_checkpoint(model, episode, score, epsilon, filename='best_model.pth'):
    """
    Simpan model weights dan metadata training ke file .pth.

    Dipanggil oleh training loop saat agent mencetak rekor score baru.

    Args:
        model (LinearQNet): Neural network yang akan disimpan
        episode (int): Nomor episode saat checkpoint dibuat
        score (int): Score rekor yang dicapai
        epsilon (float): Nilai epsilon saat ini (untuk resume training)
        filename (str): Nama file checkpoint (default: best_model.pth)
    """
    # Buat folder models/ jika belum ada
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    filepath = os.path.join(config.MODEL_DIR, filename)

    # Simpan state_dict (weights) + metadata training
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'episode': episode,
        'score': score,
        'epsilon': epsilon,
    }

    torch.save(checkpoint, filepath)
    print(f"[CHECKPOINT] Model tersimpan: {filepath} "
          f"(episode={episode}, score={score}, epsilon={epsilon:.4f})")


def load_checkpoint(model, filename='best_model.pth'):
    """
    Load model weights dan metadata dari file checkpoint.

    Digunakan untuk:
    1. Resume training dari titik terakhir
    2. Load model terbaik untuk mode --watch

    Args:
        model (LinearQNet): Neural network yang akan di-load weights-nya
        filename (str): Nama file checkpoint (default: best_model.pth)

    Returns:
        dict: Metadata training (episode, score, epsilon), atau None jika file tidak ada
    """
    filepath = os.path.join(config.MODEL_DIR, filename)

    if not os.path.exists(filepath):
        print(f"[CHECKPOINT] File tidak ditemukan: {filepath}")
        return None

    checkpoint = torch.load(filepath, weights_only=True)

    # Load weights ke model
    model.load_state_dict(checkpoint['model_state_dict'])

    metadata = {
        'episode': checkpoint.get('episode', 0),
        'score': checkpoint.get('score', 0),
        'epsilon': checkpoint.get('epsilon', config.EPSILON_END),
    }

    print(f"[CHECKPOINT] Model dimuat: {filepath} "
          f"(episode={metadata['episode']}, score={metadata['score']}, "
          f"epsilon={metadata['epsilon']:.4f})")

    return metadata
