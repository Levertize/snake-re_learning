# =============================================================================
# test_profiling.py — Profiling bottleneck training loop
#
# Mengukur waktu yang dihabiskan untuk setiap komponen training:
#   1. game.play_step()          — game logic + rendering
#   2. agent.train_short_memory() — online learning (1 sample)
#   3. agent.train_long_memory()  — batch training dari replay buffer
#
# Usage: python test_profiling.py
# =============================================================================

import sys
import os
import time

sys.path.append(os.path.dirname(__file__))
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from game.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
import config


def profile_training(num_episodes=50):
    """
    Jalankan training dengan timing detail per komponen.

    Args:
        num_episodes (int): Jumlah episode untuk profiling
    """
    print("=" * 60)
    print("  PROFILING — Training Loop Bottleneck Analysis")
    print(f"  Episodes: {num_episodes}")
    print("=" * 60)
    print()

    game = SnakeGame(render=False)
    agent = DQNAgent()

    # Akumulasi waktu per komponen
    time_play_step = 0.0
    time_train_short = 0.0
    time_train_long = 0.0
    time_get_action = 0.0
    time_get_state = 0.0
    time_remember = 0.0

    total_steps = 0
    total_long_trains = 0

    start_total = time.time()

    for episode in range(1, num_episodes + 1):
        state = game.reset()
        done = False

        while not done:
            # Timing: get_action
            t0 = time.perf_counter()
            action = agent.get_action(state)
            time_get_action += time.perf_counter() - t0

            # Timing: play_step
            t0 = time.perf_counter()
            reward, done, score = game.play_step(action)
            time_play_step += time.perf_counter() - t0

            # Timing: get_state
            t0 = time.perf_counter()
            next_state = game.get_state()
            time_get_state += time.perf_counter() - t0

            # Timing: remember
            t0 = time.perf_counter()
            agent.remember(state, action, reward, next_state, done)
            time_remember += time.perf_counter() - t0

            # Timing: train_short_memory
            t0 = time.perf_counter()
            agent.train_short_memory(state, action, reward, next_state, done)
            time_train_short += time.perf_counter() - t0

            state = next_state
            total_steps += 1

        # Timing: train_long_memory
        t0 = time.perf_counter()
        agent.train_long_memory()
        time_train_long += time.perf_counter() - t0
        total_long_trains += 1

        agent.decay_epsilon()
        agent.n_games += 1

    elapsed_total = time.time() - start_total

    # --- Report ---
    print(f"Total time:       {elapsed_total:.3f}s")
    print(f"Total steps:      {total_steps}")
    print(f"Total episodes:   {num_episodes}")
    print(f"Avg steps/episode:{total_steps / num_episodes:.1f}")
    print()

    # Tabel timing
    components = [
        ("game.play_step()", time_play_step),
        ("agent.get_action()", time_get_action),
        ("agent.get_state()", time_get_state),
        ("agent.remember()", time_remember),
        ("agent.train_short_memory()", time_train_short),
        ("agent.train_long_memory()", time_train_long),
    ]

    # Hitung total tracked time
    total_tracked = sum(t for _, t in components)

    print(f"{'Component':<30s} {'Time (s)':>10s} {'% Total':>10s} {'Per call (ms)':>14s}")
    print("-" * 66)

    for name, t in components:
        pct = (t / total_tracked * 100) if total_tracked > 0 else 0

        # Hitung per-call time
        if name == "agent.train_long_memory()":
            calls = total_long_trains
        else:
            calls = total_steps
        per_call_ms = (t / calls * 1000) if calls > 0 else 0

        print(f"{name:<30s} {t:>10.3f} {pct:>9.1f}% {per_call_ms:>13.3f}")

    print("-" * 66)
    print(f"{'TOTAL TRACKED':<30s} {total_tracked:>10.3f}")
    print(f"{'OVERHEAD (untracked)':<30s} {elapsed_total - total_tracked:>10.3f}")
    print()

    # Highlight bottleneck
    bottleneck_name, bottleneck_time = max(components, key=lambda x: x[1])
    bottleneck_pct = bottleneck_time / total_tracked * 100 if total_tracked > 0 else 0
    print(f">> Bottleneck: {bottleneck_name} ({bottleneck_pct:.1f}% of tracked time)")
    print()


if __name__ == '__main__':
    profile_training(num_episodes=50)
