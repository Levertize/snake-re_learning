# =============================================================================
# test_training.py — Smoke test untuk semua komponen Phase 5
# Jalankan 10 episode training secara headless untuk verifikasi integrasi
# =============================================================================

import sys
import os
sys.path.append(os.path.dirname(__file__))
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from game.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger, log_episode
import config


def test_full_training_loop():
    """Test: jalankan 10 episode training lengkap tanpa rendering."""
    logger = setup_logger('test_training')
    game = SnakeGame(render=False)
    agent = DQNAgent()

    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    num_episodes = 10

    for ep in range(num_episodes):
        state = game.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            reward, done, score = game.play_step(action)
            next_state = game.get_state()

            agent.remember(state, action, reward, next_state, done)
            agent.train_short_memory(state, action, reward, next_state, done)
            state = next_state

        agent.train_long_memory()
        agent.decay_epsilon()
        agent.n_games += 1

        scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        mean_scores.append(mean_score)

        if score > record:
            record = score
            save_checkpoint(agent.model, agent.n_games, record, agent.epsilon)

        log_episode(logger, agent.n_games, score, record, mean_score,
                    agent.epsilon, len(agent.memory))

    print(f"\n[PASS] {num_episodes} episodes selesai!")
    print(f"  Scores: {scores}")
    print(f"  Record: {record}")
    print(f"  Final mean: {mean_scores[-1]:.2f}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Memory size: {len(agent.memory)}")


def test_checkpoint_save_load():
    """Test: save dan load checkpoint."""
    from agent.model import LinearQNet

    # Buat model dan simpan
    model1 = LinearQNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    save_checkpoint(model1, episode=100, score=25, epsilon=0.5, filename='test_model.pth')

    # Buat model baru dan load
    model2 = LinearQNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    metadata = load_checkpoint(model2, filename='test_model.pth')

    assert metadata is not None, "Metadata harus ada setelah load"
    assert metadata['episode'] == 100
    assert metadata['score'] == 25
    assert abs(metadata['epsilon'] - 0.5) < 1e-6

    # Hapus file test
    os.remove(os.path.join(config.MODEL_DIR, 'test_model.pth'))

    print("[PASS] Checkpoint save/load — metadata benar")


def test_logger():
    """Test: logger menulis ke file."""
    logger = setup_logger('test_logger')
    log_episode(logger, 1, 5, 5, 5.0, 0.99, 100)

    # Cek folder logs/ ada dan berisi file
    assert os.path.exists(config.LOG_DIR), "Folder logs/ harus ada"
    log_files = [f for f in os.listdir(config.LOG_DIR) if f.endswith('.log')]
    assert len(log_files) > 0, "Harus ada file .log"
    print(f"[PASS] Logger — {len(log_files)} log file(s) ditemukan di {config.LOG_DIR}")


if __name__ == '__main__':
    print("=" * 50)
    print("  TRAINING INFRASTRUCTURE — Smoke Tests")
    print("=" * 50)
    print()

    test_logger()
    test_checkpoint_save_load()

    print()
    test_full_training_loop()

    print()
    print("=" * 50)
    print("  ALL TESTS PASSED!")
    print("=" * 50)
