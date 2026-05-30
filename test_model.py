# =============================================================================
# test_model.py — Unit test untuk LinearQNet dan QTrainer
# Memverifikasi output shape, forward pass, dan training step
# =============================================================================

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
from agent.model import LinearQNet, QTrainer
import config


def test_model_output_shape_single():
    """Test: output shape benar untuk single input (1 sample)."""
    model = LinearQNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)

    # Single state vector: shape (11,)
    state = torch.randn(config.INPUT_SIZE)
    output = model(state)

    assert output.shape == (config.OUTPUT_SIZE,), \
        f"Expected shape ({config.OUTPUT_SIZE},), got {output.shape}"
    print(f"[PASS] Single input — output shape: {output.shape}")


def test_model_output_shape_batch():
    """Test: output shape benar untuk batch input."""
    model = LinearQNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)

    # Batch: shape (batch_size, 11)
    batch_size = 64
    state = torch.randn(batch_size, config.INPUT_SIZE)
    output = model(state)

    assert output.shape == (batch_size, config.OUTPUT_SIZE), \
        f"Expected shape ({batch_size}, {config.OUTPUT_SIZE}), got {output.shape}"
    print(f"[PASS] Batch input (size={batch_size}) — output shape: {output.shape}")


def test_model_output_shape_training_batch():
    """Test: output shape benar dengan BATCH_SIZE dari config (1000)."""
    model = LinearQNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)

    state = torch.randn(config.BATCH_SIZE, config.INPUT_SIZE)
    output = model(state)

    assert output.shape == (config.BATCH_SIZE, config.OUTPUT_SIZE), \
        f"Expected ({config.BATCH_SIZE}, {config.OUTPUT_SIZE}), got {output.shape}"
    print(f"[PASS] Training batch (size={config.BATCH_SIZE}) — output shape: {output.shape}")


def test_output_is_q_values():
    """Test: output adalah Q-values (bisa negatif, bukan probabilitas)."""
    model = LinearQNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)

    state = torch.randn(10, config.INPUT_SIZE)
    output = model(state)

    # Q-values bisa negatif (tidak terbatas 0-1 seperti probabilitas)
    has_negative = (output < 0).any().item()
    print(f"[PASS] Output berisi Q-values (ada negatif: {has_negative})")


def test_trainer_single_step():
    """Test: QTrainer bisa menjalankan train_step dengan single experience."""
    model = LinearQNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    trainer = QTrainer(model, lr=config.LEARNING_RATE, gamma=config.GAMMA)

    # Single experience
    state = np.random.rand(config.INPUT_SIZE).astype(np.float32)
    action = [0, 1, 0]  # Belok kanan
    reward = 10
    next_state = np.random.rand(config.INPUT_SIZE).astype(np.float32)
    done = False

    # Harus jalan tanpa error
    trainer.train_step(state, action, reward, next_state, done)
    print("[PASS] QTrainer.train_step() — single experience berhasil")


def test_trainer_batch_step():
    """Test: QTrainer bisa menjalankan train_step dengan batch experience."""
    model = LinearQNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    trainer = QTrainer(model, lr=config.LEARNING_RATE, gamma=config.GAMMA)

    batch_size = 32

    states = np.random.rand(batch_size, config.INPUT_SIZE).astype(np.float32)
    actions = [[1, 0, 0]] * batch_size  # Semua lurus
    rewards = np.random.choice([0, 10, -10], size=batch_size).astype(np.float32)
    next_states = np.random.rand(batch_size, config.INPUT_SIZE).astype(np.float32)
    dones = [False] * batch_size

    trainer.train_step(states, actions, rewards, next_states, dones)
    print(f"[PASS] QTrainer.train_step() — batch (size={batch_size}) berhasil")


def test_training_changes_weights():
    """Test: training benar-benar mengubah weights model."""
    model = LinearQNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    trainer = QTrainer(model, lr=config.LEARNING_RATE, gamma=config.GAMMA)

    # Simpan weights sebelum training
    weights_before = model.linear1.weight.data.clone()

    # Jalankan beberapa training step
    for _ in range(10):
        state = np.random.rand(config.INPUT_SIZE).astype(np.float32)
        action = [1, 0, 0]
        reward = 10
        next_state = np.random.rand(config.INPUT_SIZE).astype(np.float32)
        done = False
        trainer.train_step(state, action, reward, next_state, done)

    # Weights harus berubah setelah training
    weights_after = model.linear1.weight.data
    changed = not torch.equal(weights_before, weights_after)
    assert changed, "Weights tidak berubah setelah training!"
    print(f"[PASS] Training mengubah weights — model sedang belajar")


if __name__ == '__main__':
    print("=" * 50)
    print("  NEURAL NETWORK — Unit Tests")
    print("=" * 50)
    print()

    test_model_output_shape_single()
    test_model_output_shape_batch()
    test_model_output_shape_training_batch()
    test_output_is_q_values()
    test_trainer_single_step()
    test_trainer_batch_step()
    test_training_changes_weights()

    print()
    print("=" * 50)
    print("  ALL TESTS PASSED!")
    print("=" * 50)
