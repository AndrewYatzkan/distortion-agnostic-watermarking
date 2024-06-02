from encoder import Encoder
from decoder import Decoder
from attack import AttackNetwork

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import io
import argparse

# Define the VGG-based perceptual loss
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg16().features
        self.layers = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()
    
    def forward(self, x, y):
        x_vgg = self.layers(x)
        y_vgg = self.layers(y)
        loss = self.criterion(x_vgg, y_vgg)
        return loss

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, encoder, decoder, attack_network, optimizer_enc_dec, optimizer_adv):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        attack_network.load_state_dict(checkpoint['attack_network_state_dict'])
        optimizer_enc_dec.load_state_dict(checkpoint['optimizer_enc_dec_state_dict'])
        optimizer_adv.load_state_dict(checkpoint['optimizer_adv_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_enc_dec = checkpoint['loss_enc_dec']
        loss_adv = checkpoint['loss_adv']
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        return start_epoch, loss_enc_dec, loss_adv
    else:
        print(f"No checkpoint found at '{checkpoint_path}', starting from scratch")
        return 0, None, None

def jpeg_transformation(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=50)
    buffer.seek(0)
    return Image.open(buffer)

# Define the training function
def train_model(encoder, decoder, attack_network, dataloader, optimizer_enc_dec, optimizer_adv, criterion_img, criterion_msg, alphaI1, alphaI2, alphaM, alphaAdv1, alphaAdv2, alphaAdvW, num_iter, num_epochs, device, config, save_path, transformations=None, transform_name=None):
    if transformations and transform_name:
        transform = transformations[transform_name]
    else:
        transform = None

    vgg_loss = VGGLoss().to(device)
    
    high_bit_accuracy_threshold = 0.95
    low_bit_accuracy_threshold = 0.50
    training_phase = 'enc_dec'  # Start with encoder-decoder training

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        if transform is None:
            attack_network.train()
        
        running_loss_enc_dec = 0.0
        running_loss_adv = 0.0
        running_msg_loss = 0.0
        adv_correct_bits = 0
        adv_total_bits = 0
        enc_dec_correct_bits = 0
        enc_dec_total_bits = 0
        enc_dec_adv_correct_bits = 0
        enc_dec_adv_total_bits = 0

        if training_phase == 'enc_dec' and transform_name is not None:
            transform_names = ['identity', 'jpeg', 'crop', 'dropout', 'blur', 'attack_network']
            transform_index = epoch % len(transform_names)
            new_transform = transform_names[transform_index]
            if new_transform == 'attack_network':
                transform = None
                training_phase = 'attack'
            else:
                transform = transformations[new_transform]
            print(f"transform: {new_transform}")


        for i, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = images.to(device)
            messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], config.message_length))).to(device)

            if training_phase != 'attack':
                # --- Train the encoder-decoder ---
                optimizer_enc_dec.zero_grad()
                
                # Encode
                encoded_images = encoder(images, messages)
                clipped_encoded_images = torch.clamp(encoded_images, 0, 1) # Clip encoded images to [0, 1]

                # Decode
                # decoded_messages = decoder(clipped_encoded_images)
                decoded_messages = decoder(encoded_images)


                # Generate adversarial examples for loss calculation
                if transform:
                    adv_images = torch.stack([transform(img).to(device) for img in clipped_encoded_images])
                else:
                    adv_images = attack_network(encoded_images.detach())  # Use adversarial attack
                    adv_images = attack_network(clipped_encoded_images.detach())

                decoded_adv_messages = decoder(adv_images)
                
                
                predicted_bits = (decoded_messages > 0).float()
                enc_dec_correct_bits += (predicted_bits == messages).sum().item()
                enc_dec_total_bits += messages.numel()

                predicted_bits = (decoded_adv_messages > 0).float()
                enc_dec_adv_correct_bits += (predicted_bits == messages).sum().item()
                enc_dec_adv_total_bits += messages.numel()

                loss_img = alphaI1 * criterion_img(encoded_images, images) + alphaI2 * vgg_loss(encoded_images, images)
                
                # Calculate message loss
                loss_msg = alphaM * criterion_msg(decoded_messages, messages)
                
                # Calculate the additional term involving adversarial examples
                adv_term = alphaAdvW * criterion_msg(decoded_adv_messages, messages)
                
                # Total watermarking loss
                loss_w = loss_img + loss_msg + adv_term
                
                loss_w.backward(retain_graph=True)

                optimizer_enc_dec.step()
                
                running_loss_enc_dec += loss_w.item() * images.size(0)
                running_msg_loss += loss_msg.item() * images.size(0)

            elif training_phase == 'attack':
                # --- Train the attack network ---
                for x in range(num_iter):
                    optimizer_adv.zero_grad()
                    
                    # Generate adversarial examples
                    adv_images = attack_network(encoded_images.detach())
                    decoded_adv_messages = decoder(adv_images)
                    
                    # Calculate adversarial loss
                    perturbation_loss = alphaAdv1 * torch.norm(adv_images - encoded_images.detach(), p=2)
                    adv_message_loss = alphaAdv2 * criterion_msg(decoded_adv_messages, messages)

                    loss_adv = perturbation_loss - adv_message_loss
                    loss_adv = torch.clamp(loss_adv, -100, 100)
                    
                    loss_adv.backward()

                    optimizer_adv.step()
                    
                    predicted_bits = (decoded_adv_messages > 0).float()
                    adv_correct_bits += (predicted_bits == messages).sum().item()
                    adv_total_bits += messages.numel()

                    running_loss_adv += loss_adv.item() * images.size(0)
        
        epoch_loss_enc_dec = running_loss_enc_dec / len(dataloader.dataset)
        epoch_loss_adv = running_loss_adv / len(dataloader.dataset)
        epoch_msg_loss = running_msg_loss / len(dataloader.dataset)

        adv_total_bits = adv_total_bits if adv_total_bits > 0 else 1
        enc_dec_total_bits = enc_dec_total_bits if enc_dec_total_bits > 0 else 1
        enc_dec_adv_total_bits = enc_dec_adv_total_bits if enc_dec_adv_total_bits > 0 else 1
        adv_bit_accuracy = adv_correct_bits / adv_total_bits
        enc_dec_bit_accuracy = enc_dec_correct_bits / enc_dec_total_bits
        enc_dec_adv_bit_accuracy = enc_dec_adv_correct_bits / enc_dec_adv_total_bits
        
        print(f"Epoch {epoch+1}/{num_epochs}, Watermarking Loss: {epoch_loss_enc_dec:.4f}, Adversarial Loss: {epoch_loss_adv:.4f}, Adv Bit Accuracy: {adv_bit_accuracy:.4f}, Enc/Dec Bit Accuracy: {enc_dec_bit_accuracy:.4f} Enc/Dec/Adv Bit Accuracy: {enc_dec_adv_bit_accuracy:.4f}, Phase: {training_phase}")

        # Check if we need to switch training phase
        if training_phase != 'attack' and transform is None and enc_dec_bit_accuracy >= high_bit_accuracy_threshold and enc_dec_adv_bit_accuracy >= high_bit_accuracy_threshold:
            training_phase = 'attack' if training_phase == 'enc_dec' else 'enc_dec'
            print(f"Switching to {training_phase} network training at epoch {epoch+1}")
        elif training_phase == 'attack' and adv_bit_accuracy <= low_bit_accuracy_threshold:
            training_phase = 'enc_dec-switched'
            print(f"Switching to encoder-decoder training at epoch {epoch+1}")

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'attack_network_state_dict': attack_network.state_dict(),
            'optimizer_enc_dec_state_dict': optimizer_enc_dec.state_dict(),
            'optimizer_adv_state_dict': optimizer_adv.state_dict(),
            'loss_enc_dec': epoch_loss_enc_dec,
            'loss_adv': epoch_loss_adv,
        }, filename=os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth.tar"))

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity_path", type=str, default=None, help="Path to the identity model checkpoint")
    args = parser.parse_args()

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
    }

    
    # Check for available device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create the models
    encoder = Encoder(config).to(device)
    decoder = Decoder(config).to(device)
    attack_network = AttackNetwork(config).to(device)
    
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((config.H, config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    dataset = datasets.ImageFolder("./data/train_2014", data_transforms['train'])

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    # # Uncomment to train on subset
    # indices = np.random.choice(len(dataset), 256, replace=True)
    # subset = torch.utils.data.Subset(dataset, indices)
    # dataloader = DataLoader(subset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    # Set up the optimizers and loss functions
    params_enc_dec = list(encoder.parameters()) + list(decoder.parameters())
    optimizer_enc_dec = optim.Adam(params_enc_dec, lr=0.001)
    optimizer_adv = optim.Adam(attack_network.parameters(), lr=0.001)
    
    criterion_img = nn.MSELoss()  # Image loss
    criterion_msg = nn.BCEWithLogitsLoss()  # Message loss (binary messages)
    
    num_epochs = 1000

    # Directory to save checkpoints
    save_path = "checkpoints"
    os.makedirs(save_path, exist_ok=True)
    print(f"save_path: {save_path}")

    if not args.identity_path:
        # Load identity model checkpoint
        print("No identity model path given so training an identity model from scratch.")
        # Identity model training parameters
        alphaI1 = 6.0
        alphaI2 = 0.01
        alphaM = 1.0
        alphaAdv1 = 0
        alphaAdv2 = 0
        alphaAdvW = 0
        num_iter = 0
        train_model(encoder, decoder, attack_network, dataloader, optimizer_enc_dec, optimizer_adv, criterion_img, criterion_msg, alphaI1, alphaI2, alphaM, alphaAdv1, alphaAdv2, alphaAdvW, num_iter, num_epochs, device, config, save_path, transformations)
    else:
        # No identity model provided, regular training
        print(f"Identity model provided ({args.identity_path}). Using as baseline for regular training.")
        start_epoch, _, _ = load_checkpoint(args.identity_path, encoder, decoder, attack_network, optimizer_enc_dec, optimizer_adv)
        # Regular training parameters
        alphaI1 = 18.0
        alphaI2 = 0.01
        alphaM = 0.3
        alphaAdv1 = 15.0
        alphaAdv2 = 1.0
        alphaAdvW = 0.2
        num_iter = 5
        train_model(encoder, decoder, attack_network, dataloader, optimizer_enc_dec, optimizer_adv, criterion_img, criterion_msg, alphaI1, alphaI2, alphaM, alphaAdv1, alphaAdv2, alphaAdvW, num_iter, num_epochs, device, config, save_path, transformations, '')