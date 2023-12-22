import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalCritic(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        self.image_shape = image_shape

        self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x)


if __name__ == "__main__":
    # Example usage
    image_size = 16  # example image size
    channels = 1     # example number of channels
    image_shape = (channels, image_size, image_size)

    critic = ClassicalCritic(image_shape)
    # Example input tensor
    example_input = torch.randn(1, *image_shape)
    # Forward pass through the critic
    output = critic(example_input)
    print(output)
