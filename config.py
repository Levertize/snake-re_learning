# =============================================================================
# config.py — Semua hyperparameter & konfigurasi dipusatkan di sini
# Mengubah nilai di sini akan mempengaruhi seluruh sistem (game, agent, training)
# =============================================================================

# --- Game ---
GRID_SIZE     = 20      # Jumlah tile per sisi grid (20x20 = 400 tiles)
BLOCK_SIZE    = 30      # Ukuran tiap tile dalam pixel (30x30 px)
GAME_SPEED    = 0       # 0 = training turbo (tanpa delay), >0 = FPS untuk visualisasi

# --- Neural Network ---
INPUT_SIZE    = 11      # Dimensi state vector (3 danger + 4 direction + 4 food)
HIDDEN_SIZE   = 256     # Jumlah neuron di tiap hidden layer
OUTPUT_SIZE   = 3       # Jumlah aksi: [lurus, belok_kiri, belok_kanan]
LEARNING_RATE = 0.001   # Learning rate untuk optimizer Adam

# --- Reinforcement Learning ---
GAMMA         = 0.9     # Discount factor — seberapa penting reward masa depan
EPSILON_START = 1.0     # Epsilon awal — 100% explorasi (random action)
EPSILON_END   = 0.01    # Epsilon akhir — 1% explorasi (hampir pure exploitation)
EPSILON_DECAY = 0.990   # Faktor pengurangan epsilon setiap episode (lebih cepat konvergen)
MEMORY_SIZE   = 100_000 # Kapasitas replay buffer (experience yang disimpan)
BATCH_SIZE    = 1_000   # Jumlah experience per training batch

# --- Reward ---
REWARD_EAT    = +10     # Reward saat ular berhasil makan makanan
REWARD_DIE    = -10     # Penalty saat ular mati (nabrak dinding / ekor)
REWARD_STEP   = 0       # Reward per langkah biasa (tidak makan, tidak mati)

# --- Paths ---
MODEL_DIR     = "models/"   # Folder untuk menyimpan model checkpoint
LOG_DIR       = "logs/"     # Folder untuk menyimpan log training
