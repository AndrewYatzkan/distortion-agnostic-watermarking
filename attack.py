import torch
import torch.nn as nn

class AttackNetwork(nn.Module):
    def __init__(self, config):
        super(AttackNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=config.attack_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=config.attack_channels, out_channels=3, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        return x
