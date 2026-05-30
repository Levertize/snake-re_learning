# =============================================================================
# test_e2e_training.py — End-to-end training test (500 episode, headless)
#
# Menjalankan training loop lengkap tanpa rendering dan tanpa plot
# untuk evaluasi apakah agent bisa belajar dengan konfigurasi saat ini.
#
# Target: mean score >= 10 setelah 500 episode
# Usage: python test_e2e_training.py
# =============================================================================

import sys
import os
import time

sys.path.append(os.path.dirname(__file__))
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from game.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
from utils.checkpoint import save_checkpoint
import config


def run_e2e_training(num_episodes=500, print_every=50):
    """
    Jalankan training end-to-end tanpa rendering untuk evaluasi performa.

    Args:
        num_episodes (int): Jumlah episode yang dijalankan
        print_every (int): Cetak progress setiap N episode
    """
    print("=" * 60)
    print("  END-TO-END TRAINING TEST")
    print(f"  Episodes: {num_episodes}")
    print(f"  Epsilon decay: {config.EPSILON_DECAY}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Gamma: {config.GAMMA}")
    print("=" * 60)
    print()

    game = SnakeGame(render=False)
    agent = DQNAgent()

    scores = []
    total_score = 0
    record = 0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        ep_start = time.time()

        # Reset game dan jalankan episode
        state = game.reset()
        done = False
        steps = 0

        while not done:
            action = agent.get_action(state)
            reward, done, score = game.play_step(action)
            next_state = game.get_state()

            agent.remember(state, action, reward, next_state, done)
            agent.train_short_memory(state, action, reward, next_state, done)
            state = next_state
            steps += 1

        # Episode selesai — batch training
        agent.train_long_memory()
        agent.decay_epsilon()
        agent.n_games += 1

        # Tracking
        scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games

        # Checkpoint saat rekor baru
        if score > record:
            record = score
            save_checkpoint(agent.model, agent.n_games, record, agent.epsilon)

        ep_time = time.time() - ep_start

        # Progress report
        if episode % print_every == 0 or episode == 1:
            # Hitung mean score dari 50 episode terakhir (recent performance)
            recent_scores = scores[-print_every:]
            recent_mean = sum(recent_scores) / len(recent_scores)
            elapsed = time.time() - start_time

            print(
                f"[Episode {episode:>4d}/{num_episodes}] "
                f"Score: {score:>3d} | "
                f"Record: {record:>3d} | "
                f"Mean(all): {mean_score:>6.2f} | "
                f"Mean(last {len(recent_scores)}): {recent_mean:>6.2f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Memory: {len(agent.memory):>6d} | "
                f"Time: {elapsed:.0f}s"
            )

    # --- Summary ---
    elapsed_total = time.time() - start_time
    final_mean = total_score / num_episodes

    # Mean score dari 100 episode terakhir (lebih representatif)
    last_100_mean = sum(scores[-100:]) / len(scores[-100:])

    print()
    print("=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total episodes:       {num_episodes}")
    print(f"  Total time:           {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  Avg time/episode:     {elapsed_total/num_episodes:.3f}s")
    print(f"  Record (max score):   {record}")
    print(f"  Mean score (all):     {final_mean:.2f}")
    print(f"  Mean score (last 100):{last_100_mean:.2f}")
    print(f"  Final epsilon:        {agent.epsilon:.4f}")
    print(f"  Memory size:          {len(agent.memory)}")
    print()

    # Evaluasi target
    target_mean = 10
    if last_100_mean >= target_mean:
        print(f"  [PASS] TARGET TERCAPAI! Mean score (last 100) = {last_100_mean:.2f} >= {target_mean}")
    else:
        print(f"  [WARN] TARGET BELUM TERCAPAI. Mean score (last 100) = {last_100_mean:.2f} < {target_mean}")
        print(f"      Pertimbangkan: jalankan lebih banyak episode atau tuning hyperparameter.")

    print("=" * 60)

    return {
        'scores': scores,
        'record': record,
        'mean_all': final_mean,
        'mean_last_100': last_100_mean,
        'total_time': elapsed_total,
        'final_epsilon': agent.epsilon,
    }


if __name__ == '__main__':
    results = run_e2e_training(num_episodes=500, print_every=50)
