import torch.nn as nn

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            # nn.Conv2d(channels_in, channels_out, 3, stride, padding=0), # use VALID padding as per 6.1 training details
            nn.BatchNorm2d(channels_out), # GroupNorm instead?
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
