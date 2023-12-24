from torch.utils.data import Subset
from torch.utils.data import DataLoader

from dataset import GeneratedImageDataset

num_images = 1000  # Define the number of images you want to generate
dataset = GeneratedImageDataset(num_images)

# Define the indices for the subset
subset_indices = list(range(100))

# Create the subset
subset = Subset(dataset, subset_indices)

# DataLoader for training / validating / testing
subset_loader = DataLoader(subset, batch_size=32, shuffle=True)