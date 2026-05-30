# =============================================================================
# main.py — Training loop utama untuk Snake RL Agent
#
# Entry point untuk menjalankan training DQN agent.
# Menggabungkan semua komponen: Game, Agent, Plotter, Logger, Checkpoint.
#
# Alur training per episode:
#   1. Reset game → dapatkan state awal
#   2. Agent pilih aksi (epsilon-greedy)
#   3. Jalankan aksi di game → dapatkan reward, next_state, done
#   4. Simpan experience + train short memory (per step)
#   5. Jika game over: train long memory (batch), update epsilon, log
#   6. Jika score rekor baru: simpan checkpoint
#
# Usage: python main.py
# =============================================================================

from game.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
from utils.plotter import plot_training
from utils.checkpoint import save_checkpoint
from utils.logger import setup_logger, log_episode
import config


def train():
    """
    Training loop utama — jalankan episode demi episode sampai dihentikan.

    Training berjalan tanpa batas (infinite loop) dan bisa dihentikan
    kapan saja dengan Ctrl+C atau menutup window game.
    Model terbaik otomatis disimpan setiap kali ada rekor baru.
    """
    # --- Inisialisasi semua komponen ---
    logger = setup_logger()
    logger.info(f"Hyperparameters:")
    logger.info(f"  Grid: {config.GRID_SIZE}x{config.GRID_SIZE}")
    logger.info(f"  Network: {config.INPUT_SIZE} → {config.HIDDEN_SIZE} → {config.OUTPUT_SIZE}")
    logger.info(f"  LR: {config.LEARNING_RATE}, Gamma: {config.GAMMA}")
    logger.info(f"  Epsilon: {config.EPSILON_START} → {config.EPSILON_END} (decay={config.EPSILON_DECAY})")
    logger.info(f"  Batch: {config.BATCH_SIZE}, Memory: {config.MEMORY_SIZE}")
    logger.info(f"  Rewards: eat={config.REWARD_EAT}, die={config.REWARD_DIE}, step={config.REWARD_STEP}")
    logger.info(f"{'=' * 40}")

    game = SnakeGame(render=True)
    agent = DQNAgent()

    # Tracking variabel
    scores = []        # Score setiap episode
    mean_scores = []   # Rata-rata kumulatif
    total_score = 0    # Akumulasi total score
    record = 0         # Score rekor tertinggi

    logger.info("Training dimulai! Tekan Ctrl+C atau tutup window untuk berhenti.\n")

    # --- Training loop (infinite) ---
    while True:
        # 1. Reset game untuk episode baru
        state = game.reset()

        # 2. Loop per langkah dalam episode
        done = False
        while not done:
            # Agent pilih aksi berdasarkan state saat ini (epsilon-greedy)
            action = agent.get_action(state)

            # Jalankan aksi di game → dapatkan feedback
            reward, done, score = game.play_step(action)

            # Dapatkan state baru setelah aksi
            next_state = game.get_state()

            # Simpan experience ke replay buffer
            agent.remember(state, action, reward, next_state, done)

            # Train dari experience terbaru (online learning, 1 sample)
            agent.train_short_memory(state, action, reward, next_state, done)

            # Update state untuk langkah berikutnya
            state = next_state

        # --- Episode selesai (ular mati) ---

        # 3. Train dari replay buffer (batch training)
        agent.train_long_memory()

        # 4. Update epsilon (kurangi explorasi secara bertahap)
        agent.decay_epsilon()

        # 5. Update tracking
        agent.n_games += 1
        scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        mean_scores.append(mean_score)

        # 6. Cek rekor baru → simpan checkpoint
        if score > record:
            record = score
            save_checkpoint(
                model=agent.model,
                episode=agent.n_games,
                score=record,
                epsilon=agent.epsilon
            )

        # 7. Log episode
        log_episode(
            logger=logger,
            episode=agent.n_games,
            score=score,
            record=record,
            mean_score=mean_score,
            epsilon=agent.epsilon,
            memory_size=len(agent.memory)
        )

        # 8. Update live plot
        plot_training(scores, mean_scores)


if __name__ == '__main__':
    train()
