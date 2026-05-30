# =============================================================================
# test_agent.py — Test untuk ReplayBuffer dan DQNAgent
# Memverifikasi semua fungsi inti agent sebelum lanjut ke training loop
# =============================================================================

import sys
import os
sys.path.append(os.path.dirname(__file__))
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
from agent.replay_buffer import ReplayBuffer
from agent.dqn_agent import DQNAgent
from game.snake_game import SnakeGame
import config


# === REPLAY BUFFER TESTS ===

def test_buffer_push():
    """Test: push() menyimpan experience ke buffer."""
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0, "Buffer harus kosong saat init"

    buf.push(("state", "action", 10, "next_state", False))
    assert len(buf) == 1, f"Expected len=1, got {len(buf)}"
    print("[PASS] ReplayBuffer.push() — berhasil simpan experience")


def test_buffer_maxlen():
    """Test: buffer auto-drop experience lama saat penuh."""
    buf = ReplayBuffer(capacity=5)

    for i in range(10):
        buf.push((f"state_{i}", "action", i, f"next_{i}", False))

    assert len(buf) == 5, f"Expected len=5 (maxlen), got {len(buf)}"

    # Experience pertama (state_0) harus sudah di-drop
    states = [exp[0] for exp in buf.memory]
    assert "state_0" not in states, "state_0 harus sudah di-drop"
    assert "state_9" in states, "state_9 harus masih ada"
    print("[PASS] ReplayBuffer maxlen — auto-drop berjalan")


def test_buffer_sample():
    """Test: sample() mengambil random batch dari buffer."""
    buf = ReplayBuffer(capacity=100)

    for i in range(50):
        buf.push((f"state_{i}", "action", i, f"next_{i}", False))

    sample = buf.sample(10)
    assert len(sample) == 10, f"Expected sample size 10, got {len(sample)}"

    # Setiap item harus berupa tuple dengan 5 elemen
    for exp in sample:
        assert len(exp) == 5, f"Experience harus 5 elemen, got {len(exp)}"

    print("[PASS] ReplayBuffer.sample() — random batch berhasil")


# === DQN AGENT TESTS ===

def test_agent_init():
    """Test: DQNAgent berhasil diinisialisasi dengan komponen lengkap."""
    agent = DQNAgent()

    assert agent.n_games == 0, "n_games harus 0 saat init"
    assert agent.epsilon == config.EPSILON_START, "Epsilon harus EPSILON_START"
    assert agent.gamma == config.GAMMA, "Gamma harus sesuai config"
    assert len(agent.memory) == 0, "Memory harus kosong saat init"
    print(f"[PASS] DQNAgent init — epsilon={agent.epsilon}, gamma={agent.gamma}")


def test_agent_get_action():
    """Test: get_action() mengembalikan one-hot action [1,0,0] / [0,1,0] / [0,0,1]."""
    agent = DQNAgent()
    state = np.random.rand(config.INPUT_SIZE).astype(np.float32)

    action = agent.get_action(state)

    assert len(action) == 3, f"Action harus 3 elemen, got {len(action)}"
    assert sum(action) == 1, f"Action harus one-hot (sum=1), got sum={sum(action)}"
    assert action.count(1) == 1, "Harus ada tepat satu 1 di action"
    print(f"[PASS] get_action() — action: {action} (one-hot valid)")


def test_agent_remember():
    """Test: remember() menyimpan experience ke buffer."""
    agent = DQNAgent()

    state = np.random.rand(config.INPUT_SIZE).astype(np.float32)
    action = [1, 0, 0]
    reward = 10
    next_state = np.random.rand(config.INPUT_SIZE).astype(np.float32)
    done = False

    agent.remember(state, action, reward, next_state, done)
    assert len(agent.memory) == 1, f"Memory harus 1, got {len(agent.memory)}"
    print("[PASS] remember() — experience tersimpan di buffer")


def test_agent_train_short_memory():
    """Test: train_short_memory() berjalan tanpa error."""
    agent = DQNAgent()

    state = np.random.rand(config.INPUT_SIZE).astype(np.float32)
    action = [0, 1, 0]
    reward = -10
    next_state = np.random.rand(config.INPUT_SIZE).astype(np.float32)
    done = True

    agent.train_short_memory(state, action, reward, next_state, done)
    print("[PASS] train_short_memory() — training 1 step berhasil")


def test_agent_train_long_memory():
    """Test: train_long_memory() berjalan dengan data di buffer."""
    agent = DQNAgent()

    # Isi buffer dengan beberapa experience
    for _ in range(50):
        state = np.random.rand(config.INPUT_SIZE).astype(np.float32)
        action = [0, 0, 1]
        reward = 0
        next_state = np.random.rand(config.INPUT_SIZE).astype(np.float32)
        done = False
        agent.remember(state, action, reward, next_state, done)

    agent.train_long_memory()
    print(f"[PASS] train_long_memory() — batch training dari {len(agent.memory)} experiences")


def test_agent_decay_epsilon():
    """Test: decay_epsilon() mengurangi epsilon secara bertahap."""
    agent = DQNAgent()

    initial_epsilon = agent.epsilon
    agent.decay_epsilon()

    expected = initial_epsilon * config.EPSILON_DECAY
    assert abs(agent.epsilon - expected) < 1e-6, \
        f"Expected epsilon={expected}, got {agent.epsilon}"

    # Pastikan epsilon tidak turun di bawah EPSILON_END
    for _ in range(10000):
        agent.decay_epsilon()
    assert agent.epsilon >= config.EPSILON_END, \
        f"Epsilon di bawah minimum: {agent.epsilon}"

    print(f"[PASS] decay_epsilon() — epsilon={agent.epsilon:.4f} (min={config.EPSILON_END})")


def test_agent_with_real_game():
    """Test: agent bisa bermain game sungguhan (end-to-end)."""
    agent = DQNAgent()
    game = SnakeGame(render=False)

    state = game.reset()
    done = False
    steps = 0

    while not done and steps < 200:
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)
        next_state = game.get_state()

        agent.remember(state, action, reward, next_state, done)
        agent.train_short_memory(state, action, reward, next_state, done)

        state = next_state
        steps += 1

    # Train long memory di akhir episode
    agent.train_long_memory()
    agent.decay_epsilon()
    agent.n_games += 1

    print(f"[PASS] End-to-end game — steps={steps}, score={score}, "
          f"epsilon={agent.epsilon:.4f}, memory={len(agent.memory)}")


if __name__ == '__main__':
    print("=" * 50)
    print("  DQN AGENT — Unit Tests")
    print("=" * 50)
    print()

    # Replay Buffer
    test_buffer_push()
    test_buffer_maxlen()
    test_buffer_sample()

    print()

    # DQN Agent
    test_agent_init()
    test_agent_get_action()
    test_agent_remember()
    test_agent_train_short_memory()
    test_agent_train_long_memory()
    test_agent_decay_epsilon()
    test_agent_with_real_game()

    print()
    print("=" * 50)
    print("  ALL TESTS PASSED!")
    print("=" * 50)
