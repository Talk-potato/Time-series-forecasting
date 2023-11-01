import torch
import torch.nn as nn

class SampleNet(nn.Module):
    def __init__(self, device, prev_len):
        super(SampleNet, self).__init__()
        self.device = device

        hidden_channels = 10

        self.conv1 = nn.Conv2d(prev_len, hidden_channels, 1, padding=0, device=device)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 1, padding=0, device=device)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 1, padding=0, device=device)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, 1, padding=0, device=device)
        self.conv5 = nn.Conv2d(hidden_channels, hidden_channels, 1, padding=0, device=device)
        self.conv_last = nn.Conv2d(hidden_channels, 1, 1, padding=0, device=device)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        out = self.conv_last(x)
        return out