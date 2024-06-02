# Distortion-Agnostic Watermarking with Deep Learning

This repository contains our implementation of the [distortion-agnostic watermarking system](https://arxiv.org/abs/2001.04580) for CS-413 @ EPFL.

## Table of Contents

- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)

## Installation

To set up the environment for this project:

1. Ensure Python 3.8+ is installed.
2. Clone the repository:
   ```
   git clone https://github.com/AndrewYatzkan/distortion-agnostic-watermarking.git
   cd distortion-agnostic-watermarking
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set the environment variable for MacOS users:
   ```
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```
   This step is necessary because certain transformations in the project require this setting to run properly on mac.

## Model Architecture

The system leverages an Encoder-Decoder architecture with an additional Attack Network designed to train against adversarial examples, enhancing robustness to unseen distortions. The Encoder and Decoder architectures are adapted from the [HiDDeN paper](https://arxiv.org/abs/1807.09937).
- **Encoder:** Consists of 64 channels with 4 blocks.
- **Decoder:** Consists of 64 channels with 7 blocks.
- **Attack Network:** A simple CNN with two convolutional layers (3 -> 16 -> 3 channels).

## Data
Download the [MS COCO 2014](https://cocodataset.org/#download) train and validation datasets and place them in the data folder. The data folder currently has one training sample and one validation sample to demonstrate the folder structure used, but make sure you provide ample images to train.

## Training

Training involves several phases:
1. **Initial Training on Identity Transformation:** Uses the least complex settings to achieve initial learning, facilitating early convergence.
2. **Rotation through Known Distortions:** Once the model has learned basic encoding and decoding, it undergoes training with known distortions (JPEG compression, random cropping, Gaussian blur, etc.) to generalize its capabilities.
3. **Adversarial Training:** When high bit accuracy is achieved, the Attack Network is aggressively trained to introduce and defend against adversarial examples, enhancing robustness.

The training script can be executed by running:
```
python train.py --identity_path ./example.pth.tar
```
For training the identity model, omit the identity_path argument:
```
python train.py
```
## Evaluation

Evaluation measures the bit accuracy of encoded and decoded messages after applying various distortions. To run the evaluation script:
```
python evaluate.py
```
This script will compute the bit accuracy for each distortion type and print out the results.