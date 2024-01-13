from torch.utils.data import Dataset
import numpy as np
import torch

from utils.image_processing import input_output_image_generator


class GeneratedImageDataset(Dataset):
    def __init__(self, num_images, dimensions=(2, 2), scaling_factor=2):
        self.num_images = num_images
        self.dimensions = dimensions
        self.scaling_factor = scaling_factor

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        input_image, output_image = input_output_image_generator(self.dimensions,
                                                                 self.scaling_factor)
        input_image = torch.tensor(input_image, dtype=torch.float32)
        output_image = torch.tensor(output_image, dtype=torch.float32)

        # the output image is the 'label' for how it is structured now
        return input_image, output_image
