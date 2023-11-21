import numpy as np
import torch

import config_a02


def input_image_generator(rows, cols):
    """
    Generates an image where one pixel is dark and the others are light.
    The position of the dark pixel is randomly chosen.

    Parameters:
    rows (int): The number of rows of images to generate.
    cols (int): The number of columns of images to generate.

    Returns:
    Numpy array, the generated image
    """

    # Create an image with all white pixels:
    img = np.ones((rows, cols))
    # Randomly choose a pixel to be dark
    dark_pixel_row = np.random.randint(0, rows)
    dark_pixel_col = np.random.randint(0, cols)
    # Change the chosen pixel to black (all color channels to 0)
    img[dark_pixel_row, dark_pixel_col] = 0

    return img


def output_image_generator(image, nrows, ncols):
    # Convert the image to a NumPy array for easier manipulation
    image_array = np.array(image)

    if image_array[0][0] == 0:
        values_range = np.linspace(0, 1, nrows)
    elif image_array[0][0] == 1:
        values_range = np.linspace(1, 0, nrows)
    else:
        raise AttributeError('Input image wrongly generated')

    # Use broadcasting for efficient assignment
    output_image = np.tile(values_range[:, np.newaxis], ncols)

    return output_image


def train_test_image_generator(input_rows: int,
                               input_cols: int,
                               output_rows: int,
                               output_cols: int):
    """Generate tuples of training set images, start (low resolution) and target (higher
    resolution)
    Target image are the same, multiplied by a factor of 2 per dimension.
    Only produces square images

    pixels_per_side (int): number of pixels per width = height in the LR image

    Returns: LR-image, HR-image
    """

    image_LR = input_image_generator(rows=input_rows, cols=input_cols)
    image_HR = output_image_generator(image=image_LR, nrows=output_rows, ncols=output_cols)

    return image_LR, image_HR


def from_probs_to_image(probs_tensor):

    normalized_probs = probs_tensor / probs_tensor.sum()
    # Rescale the probabilities to the range [-1, 1]
    max_val = torch.max(normalized_probs)
    output_probs = ((normalized_probs / max_val) - 0.5) * 2
    output_vector = output_probs.detach().numpy().flatten()
    num_pixels = config_a02.OUTPUT_ROWS * config_a02.OUTPUT_COLS
    output_vector_subset = output_vector[:num_pixels]
    output_matrix = output_vector_subset.reshape(config_a02.OUTPUT_ROWS, config_a02.OUTPUT_COLS)

    return output_matrix

