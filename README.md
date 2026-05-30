# Snake RL Agent

AI agent yang belajar bermain game Snake secara otonom menggunakan **Deep Q-Network (DQN)** dengan Experience Replay.

![Snake RL Agent Demo](demo.gif)

Agent belajar murni dari trial and error — tanpa hardcoded rules. Setelah ~300 episode training (~1 menit), agent sudah bisa bermain dengan rata-rata score 12+.

---

## Hasil Training

| Metrik | Hasil |
|---|---|
| Record (max score) | 44 |
| Mean score (last 100 game) | 21.70 |
| Training time (500 game) | ~2 menit |
| Training episodes | 500 |

Tren score per 50 episode:
```
Episode  50: mean 0.08    (masih random)
Episode 100: mean 0.42    (mulai belajar)
Episode 200: mean 3.58    (improving)
Episode 300: mean 12.08   (sudah bisa main)
Episode 400: mean 17.60   (makin bagus)
Episode 500: mean 20.52   (konsisten tinggi)
```

---

## Arsitektur

```
State (11 dim) --> Neural Network --> Q-values (3 aksi)
                   [11 -> 256 -> 256 -> 3]

State vector:
  - 3 danger signals (lurus, kanan, kiri)
  - 4 direction flags (atas, bawah, kiri, kanan)
  - 4 food location (atas, bawah, kiri, kanan)

Aksi (relatif):
  - [1,0,0] lurus
  - [0,1,0] belok kanan
  - [0,0,1] belok kiri
```

**Algoritma:** DQN dengan Experience Replay, Epsilon-Greedy exploration, dan Bellman equation untuk target Q-value.

---

## Instalasi

### Prasyarat
- Python 3.10+
- pip

### Setup
```bash
# Clone repository
git clone <repository-url>
cd snake-rl

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `pygame-ce` — Game engine (Community Edition, kompatibel dengan Python 3.14)
- `torch` — Neural network framework (PyTorch)
- `numpy` — Operasi numerik
- `matplotlib` — Live training plot

---

## Cara Penggunaan

### Training — Latih Agent

```bash
# Training default (infinite, sampai Ctrl+C)
python main.py

# Training dengan jumlah episode tertentu
python main.py --train --episodes 500

# Training turbo (tanpa delay rendering)
python main.py --train --speed 0

# Training dengan visualisasi lambat
python main.py --train --speed 10
```

Selama training:
- Window Pygame menampilkan game secara real-time
- Matplotlib chart menunjukkan progress score
- Model terbaik otomatis disimpan ke `models/best_model.pth`
- Log training disimpan ke `logs/`

### Watch — Tonton AI Bermain

```bash
# Tonton model terbaik bermain (default 15 FPS)
python main.py --watch

# Tonton dengan kecepatan lebih lambat
python main.py --watch --speed 10

# Tonton dengan kecepatan lebih cepat
python main.py --watch --speed 30
```

### Help

```bash
python main.py --help
```

---

## Struktur Project

```
snake-rl/
├── main.py                 # Entry point (--train / --watch)
├── config.py               # Semua hyperparameter
├── requirements.txt        # Dependencies
├── AGENTS.md               # Dokumentasi lengkap project
├── Rules.md                # Aturan kerja AI agent
│
├── game/
│   └── snake_game.py       # Game environment (Pygame)
│
├── agent/
│   ├── dqn_agent.py        # DQN Agent (epsilon-greedy, replay)
│   ├── model.py            # Neural network (PyTorch)
│   └── replay_buffer.py    # Experience replay buffer
│
├── utils/
│   ├── plotter.py          # Live training plot (Matplotlib)
│   ├── logger.py           # Logging ke file & terminal
│   └── checkpoint.py       # Save & load model
│
├── models/
│   └── best_model.pth      # Model terbaik (auto-generated)
│
└── logs/
    └── training_*.log      # Log per sesi training
```

---

## Hyperparameter

| Parameter | Nilai | Deskripsi |
|---|---|---|
| Grid size | 20x20 | Ukuran arena |
| Hidden size | 256 | Neuron per hidden layer |
| Learning rate | 0.001 | Adam optimizer |
| Gamma | 0.9 | Discount factor |
| Epsilon decay | 0.990 | Kecepatan transisi explore->exploit |
| Batch size | 1000 | Experience per training batch |
| Memory size | 100,000 | Kapasitas replay buffer |
| Reward (eat) | +10 | Makan makanan |
| Reward (die) | -10 | Mati (nabrak) |

---

## Testing

```bash
# Test game environment
python test_snake.py

# Test neural network
python test_model.py

# Test DQN agent
python test_agent.py

# Test training infrastructure
python test_training.py

# Profiling bottleneck
python test_profiling.py

# End-to-end training 500 episode
python test_e2e_training.py
```

---

## Teknologi

- **Python 3.10+** — Bahasa utama
- **PyTorch** — Neural network framework
- **Pygame-CE** — Game rendering
- **NumPy** — Operasi array/tensor
- **Matplotlib** — Live training visualization

---

## Lisensi

Project ini dibuat untuk tujuan edukasi dan eksperimen Reinforcement Learning.
