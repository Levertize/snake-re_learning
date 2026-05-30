# =============================================================================
# dqn_agent.py — Deep Q-Network Agent
#
# Agent yang belajar bermain Snake menggunakan algoritma DQN.
# Menggabungkan semua komponen:
#   - LinearQNet (neural network) untuk estimasi Q-values
#   - QTrainer untuk training (backpropagation)
#   - ReplayBuffer untuk experience replay
#   - Epsilon-greedy policy untuk balance explore vs exploit
#
# Interface utama:
#   - get_action(state) → pilih aksi berdasarkan epsilon-greedy
#   - remember(...)     → simpan experience ke replay buffer
#   - train_short_memory(...)  → train dari 1 experience (online learning)
#   - train_long_memory()      → train dari batch di replay buffer
# =============================================================================

import numpy as np
import random

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

from agent.model import LinearQNet, QTrainer
from agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent untuk game Snake.

    Agent ini menggunakan:
    1. Neural network (LinearQNet) untuk memprediksi Q-values tiap aksi
    2. Epsilon-greedy policy untuk balance exploration vs exploitation
    3. Experience replay untuk training yang lebih stabil
    4. Short-term memory (train per step) + long-term memory (train per episode)

    Attributes:
        n_games (int): Jumlah game/episode yang sudah dimainkan
        epsilon (float): Probabilitas explorasi (random action)
        gamma (float): Discount factor untuk future reward
        memory (ReplayBuffer): Buffer penyimpanan experience
        model (LinearQNet): Neural network untuk Q-value estimation
        trainer (QTrainer): Training logic (loss + backprop)
    """

    def __init__(self):
        """
        Inisialisasi DQN Agent dengan semua komponen.

        Semua hyperparameter diambil dari config.py:
        - Epsilon: mulai dari 1.0 (100% random), decay 0.995 per episode
        - Network: 11 → 256 → 256 → 3
        - Memory: kapasitas 100,000 experience
        - Gamma: 0.9
        """
        self.n_games = 0
        self.epsilon = config.EPSILON_START   # 1.0 — awalnya full random
        self.gamma = config.GAMMA             # 0.9

        # Replay buffer untuk experience replay
        self.memory = ReplayBuffer(config.MEMORY_SIZE)

        # Neural network: Input(11) → Hidden(256) → Hidden(256) → Output(3)
        self.model = LinearQNet(
            config.INPUT_SIZE,
            config.HIDDEN_SIZE,
            config.OUTPUT_SIZE
        )

        # Trainer: Adam optimizer + MSE loss
        self.trainer = QTrainer(
            self.model,
            lr=config.LEARNING_RATE,
            gamma=self.gamma
        )

    def get_action(self, state):
        """
        Pilih aksi menggunakan epsilon-greedy policy.

        Dengan probabilitas epsilon → aksi RANDOM (explorasi)
        Dengan probabilitas (1 - epsilon) → aksi terbaik dari network (eksploitasi)

        Epsilon di-decay setiap episode sehingga agent makin lama makin
        sedikit melakukan explorasi dan makin percaya pada network-nya.

        Args:
            state (np.ndarray): State vector 11 dimensi dari game

        Returns:
            list: Aksi one-hot [1,0,0] (lurus), [0,1,0] (kanan), [0,0,1] (kiri)
        """
        action = [0, 0, 0]

        if random.random() < self.epsilon:
            # EXPLORASI: pilih aksi random
            # Penting di awal training agar agent mengeksplorasi berbagai strategi
            move = random.randint(0, 2)
            action[move] = 1
        else:
            # EKSPLOITASI: pilih aksi dengan Q-value tertinggi
            # Network sudah cukup "pintar" untuk memberikan rekomendasi
            import torch
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)  # Shape: (3,) — Q-value per aksi
            move = prediction.argmax().item()       # Index aksi dengan Q-value max
            action[move] = 1

        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Simpan satu experience ke replay buffer.

        Experience ini akan digunakan nanti untuk train_long_memory()
        (batch training dari random sample).

        Args:
            state (np.ndarray): State saat aksi diambil
            action (list): Aksi yang diambil [one-hot]
            reward (float): Reward yang diterima
            next_state (np.ndarray): State setelah aksi
            done (bool): True jika ular mati (episode selesai)
        """
        self.memory.push((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train network dari SATU experience (online learning).

        Dipanggil setiap step dalam game — membuat agent belajar
        secara inkremental dari pengalaman terbaru.

        Ini adalah "short-term memory" — agent langsung belajar dari
        apa yang baru saja terjadi, meskipun hanya satu sample.

        Args:
            state, action, reward, next_state, done: Satu experience tuple
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        """
        Train network dari BATCH experience di replay buffer (experience replay).

        Dipanggil di akhir setiap episode (saat ular mati).

        Jika buffer memiliki >= BATCH_SIZE experience:
            → Sample random BATCH_SIZE experience dari buffer
        Jika buffer masih kurang:
            → Gunakan semua experience yang ada di buffer

        Random sampling penting untuk memutus korelasi temporal —
        tanpa ini, network akan overfitting ke pengalaman terbaru.
        """
        if len(self.memory) > config.BATCH_SIZE:
            # Cukup data — sample random dari buffer
            mini_sample = self.memory.sample(config.BATCH_SIZE)
        else:
            # Data masih sedikit — gunakan semua yang ada
            mini_sample = self.memory.sample(len(self.memory))

        # Unpack batch: pisahkan menjadi array per komponen
        # zip(*list_of_tuples) melakukan transpose:
        # [(s1,a1,r1,ns1,d1), (s2,a2,...)] → (s_all, a_all, r_all, ns_all, d_all)
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # Train network dengan seluruh batch sekaligus
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def decay_epsilon(self):
        """
        Kurangi epsilon setelah setiap episode.

        Epsilon mulai dari 1.0 (100% random) dan dikurangi dengan
        faktor EPSILON_DECAY (0.995) setiap episode, sampai minimum
        EPSILON_END (0.01).

        Ini membuat agent secara bertahap beralih dari explorasi
        (random) ke eksploitasi (menggunakan network).

        Formula: epsilon = max(epsilon * decay, epsilon_end)
        """
        self.epsilon = max(
            self.epsilon * config.EPSILON_DECAY,
            config.EPSILON_END
        )
