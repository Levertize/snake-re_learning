# =============================================================================
# main.py — Entry point untuk Snake RL Agent
#
# Dua mode utama:
#   1. Training mode (default): Latih agent dari nol atau lanjutkan training
#   2. Watch mode: Tonton model terbaik bermain Snake
#
# Usage:
#   python main.py                        # Training (default)
#   python main.py --train                # Training (eksplisit)
#   python main.py --watch                # Tonton AI bermain
#   python main.py --speed 10             # Set game speed (FPS)
#   python main.py --episodes 500         # Training N episode lalu berhenti
#   python main.py --watch --speed 10     # Tonton dengan kecepatan tertentu
# =============================================================================

import argparse

from game.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
from utils.plotter import plot_training
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger, log_episode
from agent.model import LinearQNet
import config


def train(speed=None, max_episodes=None):
    """
    Training loop utama — jalankan episode demi episode.

    Training berjalan sampai max_episodes tercapai (jika di-set),
    atau tanpa batas (infinite loop) jika max_episodes=None.
    Bisa dihentikan kapan saja dengan Ctrl+C atau menutup window game.
    Model terbaik otomatis disimpan setiap kali ada rekor baru.

    Args:
        speed (int, optional): Game speed override. None = pakai config.GAME_SPEED
        max_episodes (int, optional): Jumlah episode maksimal. None = infinite
    """
    # --- Inisialisasi semua komponen ---
    logger = setup_logger()
    logger.info(f"Hyperparameters:")
    logger.info(f"  Grid: {config.GRID_SIZE}x{config.GRID_SIZE}")
    logger.info(f"  Network: {config.INPUT_SIZE} -> {config.HIDDEN_SIZE} -> {config.OUTPUT_SIZE}")
    logger.info(f"  LR: {config.LEARNING_RATE}, Gamma: {config.GAMMA}")
    logger.info(f"  Epsilon: {config.EPSILON_START} -> {config.EPSILON_END} (decay={config.EPSILON_DECAY})")
    logger.info(f"  Batch: {config.BATCH_SIZE}, Memory: {config.MEMORY_SIZE}")
    logger.info(f"  Rewards: eat={config.REWARD_EAT}, die={config.REWARD_DIE}, step={config.REWARD_STEP}")
    logger.info(f"{'=' * 40}")

    # Override game speed jika di-set via CLI
    if speed is not None:
        config.GAME_SPEED = speed

    game = SnakeGame(render=True)
    agent = DQNAgent()

    # Tracking variabel
    scores = []        # Score setiap episode
    mean_scores = []   # Rata-rata kumulatif
    total_score = 0    # Akumulasi total score
    record = 0         # Score rekor tertinggi

    if max_episodes is not None:
        logger.info(f"Training {max_episodes} episode. Tekan Ctrl+C untuk berhenti lebih awal.\n")
    else:
        logger.info("Training dimulai! Tekan Ctrl+C atau tutup window untuk berhenti.\n")

    # --- Training loop ---
    episode = 0
    while max_episodes is None or episode < max_episodes:
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
        episode += 1
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

    # Training selesai (jika max_episodes di-set)
    if max_episodes is not None:
        logger.info(f"\nTraining selesai! {max_episodes} episode.")
        logger.info(f"Record: {record} | Mean score: {total_score / max_episodes:.2f}")


def watch(speed=15):
    """
    Watch mode — tonton model terbaik bermain Snake.

    Load model dari checkpoint, set epsilon=0 (pure exploitation),
    dan jalankan game dengan rendering tanpa training.

    Args:
        speed (int): Game speed dalam FPS (default: 15)
    """
    import pygame

    print("=" * 50)
    print("  WATCH MODE — Menonton AI bermain Snake")
    print("=" * 50)
    print()

    # Load model terbaik
    model = LinearQNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    metadata = load_checkpoint(model)

    if metadata is None:
        print("[ERROR] Tidak ada model yang tersimpan!")
        print("        Jalankan training dulu: python main.py --train")
        return

    print(f"  Model terbaik: score={metadata['score']}, episode={metadata['episode']}")
    print(f"  Game speed: {speed} FPS")
    print(f"  Tekan Ctrl+C atau tutup window untuk berhenti.")
    print()

    # Setup game dengan speed yang diminta
    config.GAME_SPEED = speed
    game = SnakeGame(render=True)

    # Buat agent dengan epsilon=0 (tanpa random, pure exploitation)
    agent = DQNAgent()
    agent.model = model
    agent.epsilon = 0  # Tidak ada explorasi — murni exploit

    game_num = 0
    total_score = 0

    while True:
        state = game.reset()
        done = False

        while not done:
            # Agent pilih aksi terbaik (epsilon=0 → selalu argmax)
            action = agent.get_action(state)

            # Jalankan aksi di game
            reward, done, score = game.play_step(action)

            # Update state
            state = game.get_state()

        # Episode selesai
        game_num += 1
        total_score += score
        mean = total_score / game_num
        print(f"  Game {game_num:>3d} | Score: {score:>3d} | "
              f"Mean: {mean:>6.2f} | Best model score: {metadata['score']}")

        # Pause sebelum game baru agar bisa lihat final state
        pygame.time.wait(1000)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Snake RL Agent — Train atau tonton AI bermain Snake',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python main.py                        Training (default, infinite)
  python main.py --train --episodes 500 Training 500 episode
  python main.py --watch                Tonton AI bermain
  python main.py --watch --speed 10     Tonton dengan kecepatan lambat
  python main.py --train --speed 0      Training turbo (tanpa delay)
        """
    )

    # Mode: mutually exclusive group
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--train', action='store_true', default=False,
        help='Mode training: latih agent (default jika tidak ada flag)'
    )
    mode_group.add_argument(
        '--watch', action='store_true', default=False,
        help='Mode watch: tonton model terbaik bermain'
    )

    # Opsi tambahan
    parser.add_argument(
        '--speed', type=int, default=None,
        help='Game speed dalam FPS (0=turbo, default: 0 untuk train, 15 untuk watch)'
    )
    parser.add_argument(
        '--episodes', type=int, default=None,
        help='Jumlah episode training (default: infinite, hanya untuk --train)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.watch:
        # Watch mode — tonton AI bermain
        watch_speed = args.speed if args.speed is not None else 15
        watch(speed=watch_speed)
    else:
        # Training mode (default)
        train(speed=args.speed, max_episodes=args.episodes)
