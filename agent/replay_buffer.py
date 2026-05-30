# =============================================================================
# replay_buffer.py — Experience Replay Memory untuk DQN
#
# Menyimpan pengalaman agent dalam bentuk tuple:
#   (state, action, reward, next_state, done)
#
# Experience Replay memutus korelasi temporal antar data training
# dengan melakukan random sampling dari buffer, sehingga neural network
# tidak overfitting ke sequence terakhir yang dialami.
#
# Implementasi menggunakan collections.deque dengan maxlen untuk
# auto-drop experience paling lama saat buffer penuh.
# =============================================================================

import random
from collections import deque


class ReplayBuffer:
    """
    Circular buffer untuk menyimpan experience (pengalaman) agent.

    Saat buffer penuh, experience paling lama otomatis di-drop (FIFO).
    Random sampling dari buffer digunakan untuk training agar menghindari
    korelasi temporal yang bisa membuat training tidak stabil.

    Attributes:
        memory (deque): Buffer penyimpanan dengan kapasitas terbatas
    """

    def __init__(self, capacity):
        """
        Inisialisasi replay buffer dengan kapasitas tertentu.

        Args:
            capacity (int): Jumlah maksimal experience yang bisa disimpan.
                           Sesuai config: 100,000
        """
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        """
        Simpan satu experience ke dalam buffer.

        Jika buffer sudah penuh, experience paling lama akan otomatis
        di-drop oleh deque (FIFO — First In, First Out).

        Args:
            experience (tuple): Tuple berisi (state, action, reward, next_state, done)
                - state (np.ndarray): State saat aksi diambil
                - action (list): Aksi yang diambil (one-hot)
                - reward (float): Reward yang diterima
                - next_state (np.ndarray): State setelah aksi
                - done (bool): True jika episode selesai
        """
        self.memory.append(experience)

    def sample(self, batch_size):
        """
        Ambil random sample dari buffer untuk training.

        Random sampling penting untuk memutus korelasi temporal —
        kalau kita train secara sequential, network akan overfitting
        ke pengalaman terbaru dan melupakan pengalaman lama.

        Args:
            batch_size (int): Jumlah experience yang di-sample

        Returns:
            list: List berisi `batch_size` experience tuples
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Jumlah experience yang saat ini tersimpan di buffer.

        Returns:
            int: Jumlah experience dalam buffer
        """
        return len(self.memory)
