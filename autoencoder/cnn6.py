from torch import nn
import torch
import numpy as np


numOfLabels = 4
maxStage = 6


# input: tensor of size (minibatchsize, numberOfChannels=1, Length=64, Width=64)
# output: tensor of the same size
class CNN(nn.Module):
    def __init__(
            self, 
            in_channels=1, 
            out_features=8, 
            depth=6
            ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()

        # create 2d convolutional layers
        self.convs = nn.ModuleList()
        for i in range(depth):
            out_channels = 16 * 2 ** i
            self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'))
            in_channels = out_channels

        # create fully connected layers
        self.fullyConnected = nn.Linear(in_features=out_channels*4, out_features=out_channels*2, bias=True)

        # create output layer
        self.output = nn.Linear(in_features=out_channels*2, out_features=out_features, bias=True)

    def forward(self, x):

        # apply convolutional layers
        for conv in self.convs:
            x = self.pool(self.relu(conv(x)))

        # flatten and apply fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fullyConnected(x))

        # apply output layer
        x = self.output(x)
        return x
    
    def test(self):
        # test that CNN-6 is working
        random_data = torch.rand((20, 1, 128, 128))
        result = self.forward(random_data)
        print(result.shape)

def WingLoss(label_pred, label_true):
    batch_size = label_pred.shape[0]
    label_size = label_pred.shape[1]
    loss = 0
    for b in range(batch_size):
        for l in range(label_size):
            loss += wing(label_pred[b, l] - label_true[b, l])
    return loss


def wing(x, w=10, eps=2):
    if abs(x) < w:
        return w * np.log(1 + abs(x) / eps)
    else:
        return abs(x) - w + w * np.log(1 + w / eps)

