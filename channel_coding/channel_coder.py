import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class ChannelEncoder(nn.Module):
    def __init__(self, input_length, layer_sizes, device_type):
        super(ChannelEncoder, self).__init__()
        self.device_type = device_type
        layers = []
        in_features = input_length
        for out_features in layer_sizes:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers[-1] = nn.Sigmoid()  # Change last activation to Sigmoid
        self.model = nn.Sequential(*layers).to(self.device_type)

    def forward(self, x):
        return self.model(x.to(self.device_type))

class ChannelDecoder(nn.Module):
    def __init__(self, input_length, layer_sizes, device_type):
        super(ChannelDecoder, self).__init__()
        self.device_type = device_type
        layers = []
        in_features = input_length
        for out_features in layer_sizes:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers[-1] = nn.Sigmoid()  # Change last activation to Sigmoid
        self.model = nn.Sequential(*layers).to(self.device_type)

    def forward(self, x):
        return self.model(x.to(self.device_type))

class MessageDataset(Dataset):
    def __init__(self, message_length):
        self.message_length = message_length

    def __len__(self):
        return 10000  # Arbitrary large number

    def __getitem__(self, idx):
        return torch.randint(0, 2, (self.message_length,)).float()

def get_device_type():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Channel Coding Model
class ChannelCodingModule(pl.LightningModule):
    def __init__(self, config):
        super(ChannelCodingModule, self).__init__()
        self.save_hyperparameters(config)
        self.device_type = get_device_type()
        self.channel_encoder = ChannelEncoder(config['input_length'], config['encoder_layer_sizes'], self.device_type)
        self.channel_decoder = ChannelDecoder(config['encoder_layer_sizes'][-1], config['decoder_layer_sizes'], self.device_type)
        self.criterion = nn.MSELoss()
        self.learning_rate = config['learning_rate']
        self.max_noise_level = config['max_noise_level']

    def forward(self, x):
        encoded_message = self.channel_encoder(x.to(self.device_type)).round()
        noisy_encoded_message = self.add_noise(encoded_message, self.max_noise_level, randomize_noise=True)
        decoded_message = self.channel_decoder(noisy_encoded_message)
        return decoded_message

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device_type)
        decoded_messages = self(batch)
        loss = self.criterion(decoded_messages, batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def generate_random_messages(batch_size, message_length):
        return torch.randint(0, 2, (batch_size, message_length)).float()

    @staticmethod
    def add_noise(encoded_message, noise_level, randomize_noise=False):
        if randomize_noise:
            noise_level = torch.rand(1).item() * noise_level  # Randomly sample from [0, noise_level]

        noise = torch.bernoulli(torch.full_like(encoded_message, noise_level))
        noisy_encoded_message = torch.abs(encoded_message - noise)  # Flip bits
        return noisy_encoded_message

if __name__ == '__main__':
    N = 16
    input_length = 5
    version = "m1-2"
    m = 1

    print(f"Training with N={N}")
    config = {
        'input_length': input_length,
        'encoder_layer_sizes': [256 * m, 512 * m, 1024 * m, N],
        'decoder_layer_sizes': [1024 * m, 512 * m, 256 * m, input_length],
        'batch_size': 1024,
        'num_epochs': 1000,
        'learning_rate': 1e-4,
        'max_noise_level': 0.3,
    }

    dataset = MessageDataset(config['input_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=9, persistent_workers=True, pin_memory=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=f'checkpoints/{input_length}_to_{N}-{version}/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    logger = TensorBoardLogger("lightning_logs", name=f"{input_length}_to_{N}", version=version)

    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        callbacks=[checkpoint_callback],
        logger=logger,
        # profiler='simple',
        log_every_n_steps=10,
    )
    model = ChannelCodingModule(config)
    trainer.fit(model, dataloader)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")