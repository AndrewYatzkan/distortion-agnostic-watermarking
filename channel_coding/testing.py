import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from channel_coder import ChannelCodingModule

# calculates the bit accuracy between the original and decoded messages
def bit_accuracy(original, decoded, device):
    original = original.round().int().to(device)
    decoded = decoded.round().int().to(device)
    return (original == decoded).float().mean().item() * 100

# loads the latest checkpoint from the specified directory
def load_latest_checkpoint(checkpoint_dir, input_length, output_length):
    latest_checkpoint = None
    latest_time = None
    for ckpt in os.listdir(checkpoint_dir):
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        if os.path.isfile(ckpt_path):
            ckpt_time = os.path.getmtime(ckpt_path)
            if latest_time is None or ckpt_time > latest_time:
                latest_checkpoint = ckpt_path
                latest_time = ckpt_time

    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    model = ChannelCodingModule.load_from_checkpoint(latest_checkpoint)
    return model

# tests the NECST model with the specified noise levels
def test_necst_model(model, input_length=30, noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4], device='cpu'):
    results = {}
    for noise_level in noise_levels:
        accuracies = []
        for _ in range(1000):  # Use a large number of samples for reliable statistics
            message = model.generate_random_messages(1, input_length)
            encoded_message = model.channel_encoder(message).round()
            noisy_encoded_message = model.add_noise(encoded_message, noise_level)
            decoded_message = model.channel_decoder(noisy_encoded_message)
            accuracy = bit_accuracy(message, decoded_message, device)
            accuracies.append(accuracy)
        results[noise_level] = np.mean(accuracies)
        print(f"Noise Level: {noise_level}, Bit Accuracy: {results[noise_level]:.4f}")
    return results

# Paths to model files
model_dirs = {
    # 'N=150': 'checkpoints/N150/',
    # 'N=120': 'checkpoints/N120/',
    # 'N=90': 'checkpoints/N90/',
    # 'N=200-2x': 'checkpoints/N200-2x-layers/'
    # 'N=150': 'checkpoints/N150/',
    # 'N=120': 'checkpoints/N120/',
    # 'N=90': 'checkpoints/N90/',
    # 'N=200-2x': 'checkpoints/N200-2x-layers/'
    # 'N=200': 'checkpoints/N200-latest-x8',
    # 'N=255': 'checkpoints/N255-latest-x8-theo0.3',
    # 'N=1040': 'checkpoints/N1040-latest-x8-theo0.4',
    'N=16': 'checkpoints/5_to_16-m1',
}

# input_length = 30
input_length = 5
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
results = {}

# Evaluate each model
for key, checkpoint_dir in model_dirs.items():
    print("loading", checkpoint_dir)
    model = load_latest_checkpoint(checkpoint_dir, input_length, int(key.split('=')[1].split('-')[0]))
    results[key] = test_necst_model(model, input_length, noise_levels, model.device_type)

# Plotting the results
plt.figure(figsize=(10, 6))

for key, accuracies in results.items():
    noise_levels = list(accuracies.keys())
    bit_accuracies = list(accuracies.values())
    plt.plot(noise_levels, bit_accuracies, marker='o', label=key)

plt.xlabel('Noise')
plt.ylabel('Bit Accuracy')
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4])
plt.yticks([50, 60, 70, 80, 90, 100])
plt.xlim(0.0 - 0.01, 0.4 + 0.01)
plt.ylim(50 - 3, 100 + 3)
plt.legend()
plt.grid(True)
plt.show()