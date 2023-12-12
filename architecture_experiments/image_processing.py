import numpy as np
import matplotlib.pyplot as plt
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

# TODO: still does not work perfectly but better

def input_output_image_generator(nrows_in, ncols_in, nrows_out, ncols_out):
    # Generate the training (input image)
    input_image = np.ones((nrows_in, ncols_in))
    black_pixel_row_in = np.random.randint(0, nrows_in)
    black_pixel_col_in = np.random.randint(0, ncols_in)
    input_image[black_pixel_row_in, black_pixel_col_in] = 0

    # Calculate the relative position of the black pixel in the input image
    relative_row = black_pixel_row_in / nrows_in
    relative_col = black_pixel_col_in / ncols_in

    # Scale the position to the output image dimensions
    black_pixel_row_out = int(relative_row * nrows_out)
    black_pixel_col_out = int(relative_col * ncols_out)

    # Initialize the output image
    output_image = np.ones((nrows_out, ncols_out))
    max_dist = np.sqrt(max(black_pixel_row_out, nrows_out - 1 - black_pixel_row_out)**2 +
                       max(black_pixel_col_out, ncols_out - 1 - black_pixel_col_out)**2)

    for row in range(nrows_out):
        for col in range(ncols_out):
            dist = np.sqrt((row - black_pixel_row_out)**2 + (col - black_pixel_col_out)**2)
            output_image[row, col] = dist / max_dist

    return input_image, output_image


def from_probs_to_image(probs_tensor):

    normalized_probs = probs_tensor / probs_tensor.sum()
    max_val = torch.max(normalized_probs)
    output_probs = (normalized_probs / max_val)
    output_vector = output_probs.detach().numpy().flatten()
    num_pixels = config.OUTPUT_ROWS * config.OUTPUT_COLS
    output_vector_subset = output_vector[:num_pixels]
    output_matrix = output_vector_subset.reshape(config.OUTPUT_ROWS, config.OUTPUT_COLS)

    return output_matrix



# ERASE FROM HERE #

# def plot_images(images, titles):
#     """
#     Plots a series of images in a single figure with individual subplots.
#
#     Parameters:
#     - images (list): A list of images to be plotted.
#     - titles (list): A list of titles for the subplots; should be the same length as images.
#
#     Each image is displayed in its own subplot with the corresponding title.
#     """
#     # The number of images
#     num_images = len(images)
#
#     # Create a figure and a set of subplots
#     fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
#
#     # If there's only one image, axs is not a list so we make it a list for consistency
#     if num_images == 1:
#         axs = [axs]
#
#     # Loop through each image and its corresponding title
#     for ax, image, title in zip(axs, images, titles):
#         ax.imshow(image, cmap='gray')
#         ax.title.set_text(title)
#         # ax.axis('off')  # Turn off axis
#
#     plt.tight_layout()  # Adjust subplots to fit into the figure area.
#     plt.show()
#
#
#
# input, output = input_output_image_generator(3,3,6,6)
# plot_images([input, output], ['in', 'out'])

