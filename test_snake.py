# =============================================================================
# test_snake.py — Script testing otomatis untuk verifikasi game logic
# Menjalankan game dengan aksi random dan memeriksa semua fungsi inti
# =============================================================================

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np

# Set environment variable agar Pygame tidak buka window (headless)
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from game.snake_game import SnakeGame
import config

def test_reset():
    """Test: reset() mengembalikan state vector 11 dimensi."""
    game = SnakeGame(render=False)
    state = game.reset()
    assert state.shape == (11,), f"Expected shape (11,), got {state.shape}"
    assert state.dtype == np.float32, f"Expected float32, got {state.dtype}"
    print("[PASS] reset() — state shape (11,) float32")

def test_play_step():
    """Test: play_step() mengembalikan (reward, done, score)."""
    game = SnakeGame(render=False)
    game.reset()
    
    action = [1, 0, 0]  # Lurus
    reward, done, score = game.play_step(action)
    
    assert isinstance(reward, (int, float)), f"Reward bukan angka: {type(reward)}"
    assert isinstance(done, bool), f"Done bukan bool: {type(done)}"
    assert isinstance(score, int), f"Score bukan int: {type(score)}"
    print(f"[PASS] play_step() — reward={reward}, done={done}, score={score}")

def test_state_values():
    """Test: get_state() mengembalikan nilai 0 atau 1 saja."""
    game = SnakeGame(render=False)
    state = game.reset()
    
    for i, val in enumerate(state):
        assert val in (0.0, 1.0), f"State[{i}] = {val}, expected 0.0 or 1.0"
    print(f"[PASS] get_state() — semua nilai 0 atau 1: {state.tolist()}")

def test_collision_detection():
    """Test: _is_collision() mendeteksi dinding dan ekor."""
    game = SnakeGame(render=False)
    game.reset()
    
    from game.snake_game import Point
    
    # Cek collision di luar grid (dinding)
    assert game._is_collision(Point(-1, 0)) == True, "Harusnya collision di x=-1"
    assert game._is_collision(Point(0, -1)) == True, "Harusnya collision di y=-1"
    assert game._is_collision(Point(game.w, 0)) == True, "Harusnya collision di x=w"
    assert game._is_collision(Point(0, game.h)) == True, "Harusnya collision di y=h"
    
    # Cek collision di dalam grid (aman)
    assert game._is_collision(Point(0, 0)) == False, "Harusnya aman di (0,0)"
    print("[PASS] _is_collision() — deteksi dinding benar")

def test_random_game():
    """Test: jalankan game dengan aksi random sampai game over."""
    game = SnakeGame(render=False)
    game.reset()
    
    actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    steps = 0
    done = False
    
    while not done:
        action = actions[np.random.randint(0, 3)]
        reward, done, score = game.play_step(action)
        steps += 1
        
        # Safety: stop setelah 10000 langkah
        if steps > 10000:
            print(f"[WARN] Game belum selesai setelah {steps} langkah, force stop")
            break
    
    print(f"[PASS] Random game selesai — steps={steps}, score={score}")

def test_multiple_episodes():
    """Test: jalankan beberapa episode berturut-turut untuk cek reset()."""
    game = SnakeGame(render=False)
    scores = []
    
    for episode in range(5):
        game.reset()
        done = False
        actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        while not done:
            action = actions[np.random.randint(0, 3)]
            reward, done, score = game.play_step(action)
        
        scores.append(score)
    
    print(f"[PASS] 5 episodes berturut — scores: {scores}")


if __name__ == '__main__':
    print("=" * 50)
    print("  SNAKE GAME — Automated Tests")
    print("=" * 50)
    print()
    
    test_reset()
    test_play_step()
    test_state_values()
    test_collision_detection()
    test_random_game()
    test_multiple_episodes()
    
    print()
    print("=" * 50)
    print("  ALL TESTS PASSED!")
    print("=" * 50)
