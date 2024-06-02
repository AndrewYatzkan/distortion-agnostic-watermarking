import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from encoder import Encoder
from decoder import Decoder
from attack import AttackNetwork
import matplotlib.pyplot as plt
from PIL import Image
import io

import warnings
warnings.filterwarnings("ignore", message=".*aten::_upsample_bilinear2d_aa.out.*MPS backend.*")

# Configuration
class Config:
    H = 128
    W = 128
    encoder_channels = 64
    encoder_blocks = 4
    decoder_channels = 64
    decoder_blocks = 7
    message_length = 16
    attack_channels = 16

config = Config()

# Check for available device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Create the models
encoder = Encoder(config).to(device)
decoder = Decoder(config).to(device)
attack_network = AttackNetwork(config).to(device)

# Load the checkpoint
checkpoint_path = "jun2_model.pth.tar"
checkpoint = torch.load(checkpoint_path)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
attack_network.load_state_dict(checkpoint['attack_network_state_dict'])

encoder.train()
decoder.train()

# Set up dataset and dataloader for validation set
data_transforms = {
    'val': transforms.Compose([
        transforms.CenterCrop((config.H, config.W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

validation_dataset = datasets.ImageFolder("./data/train_2014", data_transforms['val'])

indices = np.random.choice(len(validation_dataset), 16, replace=True)
subset = torch.utils.data.Subset(validation_dataset, indices)

validation_dataloader = DataLoader(subset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

# Function to show an image
def imshow(img, unnormalize=False):
    if unnormalize:
        img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    # print(np.min(np.transpose(npimg, (1, 2, 0))), np.max(np.transpose(npimg, (1, 2, 0))))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def jpeg_transformation(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=50)
    buffer.seek(0)
    return Image.open(buffer)

# Define transformations
transformations = {
    # Known distortions
    'identity': lambda x: x,
    'jpeg': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(jpeg_transformation),
        transforms.ToTensor()
    ]),
    'crop': transforms.RandomResizedCrop((config.H, config.W), scale=(0.965, 0.965)),
    'dropout': transforms.RandomErasing(p=1, scale=(0.3, 0.3), ratio=(1, 1)),
    'blur': transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 1.0)),

    # Unknown distortions
    'noise': lambda x: x + torch.randn_like(x) * 0.06,
    'hue': transforms.ColorJitter(hue=0.2),
    'saturation': transforms.ColorJitter(saturation=15.0),
    'resize': transforms.Resize((int(config.H * 0.7), int(config.W * 0.7))),
    'gif': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda img: img.convert("P", palette=Image.ADAPTIVE, colors=16)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((config.H, config.W)),
        transforms.ToTensor()
    ])
}

# Function to apply transformation and calculate bit accuracy for a batch
def calculate_bit_accuracy(images, messages, transform_type):
    # Apply transformation
    images = images / 2 + 0.5
    transform = transformations[transform_type]

    encoded_images = encoder(images, messages)
    encoded_images = encoded_images / 2 + 0.5
    encoded_images = torch.clamp(encoded_images, 0, 1)
    
    # Apply transformation
    transformed_images = torch.stack([transform(img).to(device) for img in encoded_images])
    transformed_images = torch.clamp(transformed_images, 0, 1)
    
    decoded_messages = decoder(transformed_images)
    decoded_bits = (decoded_messages > 0).float()  # BCE
    bit_accuracy = (decoded_bits == messages).float().mean().item()
    
    return bit_accuracy

# Calculate the average bit accuracy over the validation set for each transformation
for transform_type in transformations.keys():
    total_bit_accuracy = 0
    total_samples = 0

    for images, _ in validation_dataloader:
        images = images.to(device)

        # Generate a fixed message for the entire batch
        batch_size = images.shape[0]
        messages = torch.Tensor(np.random.choice([0, 1], (batch_size, config.message_length))).to(device)

        # Calculate bit accuracy for the batch
        bit_accuracy = calculate_bit_accuracy(images, messages, transform_type)
        total_bit_accuracy += bit_accuracy * batch_size
        total_samples += batch_size

    avg_bit_accuracy = total_bit_accuracy / total_samples

    # Print the average bit accuracy for the transformation
    print(f"Average Bit Accuracy for {transform_type} transformation over {total_samples} images: {avg_bit_accuracy * 100:.2f}%")
