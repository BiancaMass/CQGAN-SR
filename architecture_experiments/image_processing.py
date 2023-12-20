import numpy as np
import torch

import config


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

# TODO: it is not perfect, and you can think more about a definition of training images that
#  makes sense, but it will have to do for now.


def input_output_image_generator(dimensions=(2, 2), scaling_factor=2):
    """
    Generates and returns an input (LR) and output (corresponding HR) images.
    The output image is scaled by the given factor from the input image.

    Parameters:
    dimensions (tuple, optional): the number of rows and columns in the input image. Defaults to
    (2, 2).
    scaling_factor (int, optional): Scaling factor for the output image size. Defaults to 2.

    Returns:
    tuple: Input image (numpy array), Output image (numpy array).
    """
    nrows_in, ncols_in = dimensions
    nrows_out = nrows_in * scaling_factor
    ncols_out = ncols_in * scaling_factor

    # Generate the training (input image)
    input_image = np.ones((nrows_in, ncols_in))
    black_pixel_row_in = np.random.randint(0, nrows_in)
    black_pixel_col_in = np.random.randint(0, ncols_in)
    input_image[black_pixel_row_in, black_pixel_col_in] = 0

    # Scale the position to the output image dimensions (2x size)
    black_pixel_row_out = black_pixel_row_in * scaling_factor
    black_pixel_col_out = black_pixel_col_in * scaling_factor

    # Initialize the output image
    output_image = np.ones((nrows_out, ncols_out))

    max_dist = np.sqrt(max(black_pixel_row_out, nrows_out - 1 - black_pixel_row_out)**2 +
                       max(black_pixel_col_out, ncols_out - 1 - black_pixel_col_out)**2)

    for row in range(nrows_out):
        for col in range(ncols_out):
            dist = np.sqrt((row - black_pixel_row_out)**2 + (col - black_pixel_col_out)**2)
            output_image[row, col] = dist / max_dist

    return input_image, output_image


def train_test_image_generator(input_rows: int,
                               input_cols: int,
                               scaling_factor: int):
    """Generate tuples of training set images, start (low resolution) and target (higher
    resolution)

    Returns: LR-image, HR-image
    """

    image_LR, image_HR = input_output_image_generator((input_rows, input_cols), scaling_factor)

    return image_LR, image_HR


def from_probs_to_image(probs_tensor):

    normalized_probs = probs_tensor / probs_tensor.sum()
    max_val = torch.max(normalized_probs)
    output_probs = (normalized_probs / max_val)
    output_vector = output_probs.detach().numpy().flatten()
    num_pixels = config.OUTPUT_ROWS * config.OUTPUT_COLS
    output_vector_subset = output_vector[:num_pixels]
    output_matrix = output_vector_subset.reshape(config.OUTPUT_ROWS, config.OUTPUT_COLS)

    return output_matrix
