# =============================================================================
# model.py — Arsitektur Neural Network untuk DQN Agent
#
# Berisi dua class:
#   1. LinearQNet  — Feed-forward neural network untuk estimasi Q-values
#   2. QTrainer    — Training logic: loss calculation & backpropagation
#
# Arsitektur network:
#   Input(11) → Hidden1(256, ReLU) → Hidden2(256, ReLU) → Output(3, Linear)
#
# Output adalah Q-values untuk setiap aksi [lurus, kanan, kiri],
# bukan probabilitas — aksi dipilih menggunakan argmax.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


class LinearQNet(nn.Module):
    """
    Fully Connected Feed-Forward Neural Network untuk estimasi Q-values.

    Arsitektur:
        Input Layer  : 11 neuron  (state vector dari game)
        Hidden Layer 1: 256 neuron + ReLU activation
        Hidden Layer 2: 256 neuron + ReLU activation
        Output Layer : 3 neuron  (Q-value per aksi: lurus, kanan, kiri)

    Q-value merepresentasikan "expected cumulative reward" untuk setiap aksi.
    Agent memilih aksi dengan Q-value tertinggi (argmax).
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Inisialisasi layer-layer neural network.

        Args:
            input_size (int): Dimensi input (11 untuk Snake)
            hidden_size (int): Jumlah neuron di tiap hidden layer (256)
            output_size (int): Jumlah aksi/output (3 untuk Snake)
        """
        super().__init__()

        # Layer 1: Input → Hidden 1
        self.linear1 = nn.Linear(input_size, hidden_size)

        # Layer 2: Hidden 1 → Hidden 2
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        # Layer 3: Hidden 2 → Output (Q-values)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass — propagasi input melalui network.

        Alur: x → Linear1 → ReLU → Linear2 → ReLU → Linear3 (linear output)

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, input_size)
                              atau (input_size,) untuk single sample

        Returns:
            torch.Tensor: Q-values untuk setiap aksi, shape (batch_size, output_size)
        """
        # Hidden layer 1 dengan ReLU activation
        x = torch.relu(self.linear1(x))

        # Hidden layer 2 dengan ReLU activation
        x = torch.relu(self.linear2(x))

        # Output layer — linear (tanpa activation)
        # Q-values bisa negatif, jadi tidak pakai ReLU/sigmoid di sini
        x = self.linear3(x)

        return x


class QTrainer:
    """
    Training logic untuk LinearQNet menggunakan DQN algorithm.

    Menghitung loss antara predicted Q-values dan target Q-values,
    lalu melakukan backpropagation untuk update weights.

    Target Q-value dihitung menggunakan Bellman equation:
        Q_target = reward + gamma * max(Q(next_state))
        (jika episode selesai, Q_target = reward saja)

    Attributes:
        lr (float): Learning rate untuk optimizer Adam
        gamma (float): Discount factor — seberapa penting reward masa depan
        model (LinearQNet): Neural network yang di-train
        optimizer (Adam): Optimizer untuk update weights
        criterion (MSELoss): Loss function — Mean Squared Error
    """

    def __init__(self, model, lr, gamma):
        """
        Inisialisasi trainer dengan model, learning rate, dan gamma.

        Args:
            model (LinearQNet): Neural network yang akan di-train
            lr (float): Learning rate untuk Adam optimizer
            gamma (float): Discount factor (0-1)
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model

        # Adam optimizer — adaptif dan efektif untuk DQN
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # MSE Loss — mengukur jarak kuadrat antara predicted dan target Q-values
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Satu langkah training: forward pass → compute loss → backprop → update weights.

        Mendukung dua mode:
        1. Single experience: state shape (11,) — untuk train_short_memory
        2. Batch experience: state shape (batch_size, 11) — untuk train_long_memory

        Args:
            state (np.ndarray or torch.Tensor): State saat ini
            action (list or np.ndarray): Aksi yang diambil (one-hot atau index)
            reward (float or np.ndarray): Reward yang diterima
            next_state (np.ndarray or torch.Tensor): State setelah aksi
            done (bool or list[bool]): Apakah episode selesai

        Training menggunakan Bellman equation:
            Q_new = reward + gamma * max(Q(next_state))  [jika not done]
            Q_new = reward                                [jika done]
        """
        # Konversi semua input ke tensor PyTorch
        # Konversi ke numpy array dulu jika input berupa list of arrays,
        # untuk menghindari overhead konversi satu-per-satu yang lambat
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # Jika input single experience (1D), tambahkan dimensi batch
        # Contoh: shape (11,) → (1, 11) agar kompatibel dengan batch processing
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # Jadikan tuple agar bisa di-iterate

        # --- Forward pass ---
        # Prediksi Q-values untuk state saat ini: shape (batch_size, 3)
        pred = self.model(state)

        # Clone prediksi sebagai target — kita hanya akan mengubah
        # Q-value untuk aksi yang benar-benar diambil
        target = pred.clone()

        # --- Hitung target Q-values menggunakan Bellman equation (vektorisasi) ---
        # Satu forward pass untuk seluruh batch next_states sekaligus,
        # jauh lebih cepat daripada loop Python per-sample
        with torch.no_grad():
            next_q_values = self.model(next_state)            # Shape: (batch, 3)
            max_next_q = torch.max(next_q_values, dim=1)[0]   # Shape: (batch,)

        # done_tensor: True→0, False→1 untuk masking future reward
        # Jika done=True, future reward di-nol-kan (episode sudah selesai)
        done_tensor = torch.tensor(done, dtype=torch.float)

        # Bellman equation (vektorisasi):
        # Q_new = reward + gamma * max(Q(next_state)) * (1 - done)
        q_new = reward + self.gamma * max_next_q * (1 - done_tensor)

        # Update hanya Q-value untuk aksi yang diambil
        # argmax mengkonversi one-hot [0,1,0] → index 1
        action_indices = torch.argmax(action, dim=1)  # Shape: (batch,)
        target[torch.arange(len(action_indices)), action_indices] = q_new

        # --- Backpropagation ---
        self.optimizer.zero_grad()          # Reset gradient
        loss = self.criterion(target, pred) # Hitung MSE loss
        loss.backward()                     # Hitung gradient
        self.optimizer.step()               # Update weights
