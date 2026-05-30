# =============================================================================
# plotter.py — Live training plot menggunakan Matplotlib
#
# Menampilkan dua grafik real-time yang di-update setiap akhir episode:
#   1. Score per game — melihat performa tiap episode
#   2. Mean score — moving average untuk melihat tren keseluruhan
#
# Menggunakan plt.ion() (interactive mode) agar plot bisa di-update
# tanpa memblokir training loop.
# =============================================================================

import matplotlib.pyplot as plt
from IPython import display


plt.ion()  # Aktifkan interactive mode — plot bisa di-update tanpa blocking


def plot_training(scores, mean_scores):
    """
    Update live plot dengan data training terbaru.

    Dipanggil setiap akhir episode oleh training loop di main.py.
    Plot akan menampilkan:
    - Garis biru: score per game (fluktuatif)
    - Garis oranye: rata-rata score (tren keseluruhan)

    Args:
        scores (list[int]): Daftar score dari setiap episode
        mean_scores (list[float]): Daftar rata-rata score kumulatif
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.clf()  # Bersihkan figure sebelum menggambar ulang

    plt.title('Snake RL Agent — Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot score per game (biru)
    plt.plot(scores, label='Score', color='#2196F3', alpha=0.6)

    # Plot mean score (oranye, lebih tebal untuk emphasis)
    plt.plot(mean_scores, label='Mean Score', color='#FF9800', linewidth=2)

    # Tampilkan nilai terbaru di ujung garis
    if len(scores) > 0:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]),
                 fontsize=9, color='#2196F3')
        plt.text(len(mean_scores) - 1, mean_scores[-1],
                 f'{mean_scores[-1]:.1f}',
                 fontsize=9, color='#FF9800')

    plt.ylim(ymin=0)  # Score tidak bisa negatif
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.pause(0.1)  # Pause singkat agar plot ter-render
