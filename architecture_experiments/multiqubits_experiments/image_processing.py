import numpy as np
import torch


def image_generator(rows, cols):
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


def enlarge_image(image, scale_factor):
    # Convert the image to a NumPy array for easier manipulation
    image_array = np.array(image)

    # Repeat each element in the columns by the scale factor
    enlarged_image = np.repeat(image_array, scale_factor, axis=1)

    # Repeat each element in the rows by the scale factor
    enlarged_image = np.repeat(enlarged_image, scale_factor, axis=0)

    return enlarged_image


def train_test_image_generator(original_px_per_side: int, scale_factor: int = 2):
    """Generate touples of training set images, start (low resolution) and target (higher
    resolution)
    Target image are the same, multiplied by a factor of 2 per dimension.
    Only produces square images

    pixels_per_side (int): number of pixels per width = height in the LR image

    Returns: LR-image, HR-image
    """

    image_LR = image_generator(rows=original_px_per_side, cols=original_px_per_side)
    image_HR = enlarge_image(image=image_LR, scale_factor=scale_factor)

    return image_LR, image_HR


def from_probs_to_image(probs_tensor):

    normalized_probs = probs_tensor / probs_tensor.sum()
    # Rescale the probabilities to the range [-1, 1]
    max_val = torch.max(normalized_probs)
    output_probs = ((normalized_probs / max_val) - 0.5) * 2
    output_vector = output_probs.detach().numpy().flatten()
    output_vector_subset = output_vector[:16]  # TODO: hard coded
    pixels_side = int(np.sqrt(len(output_vector_subset)))
    output_matrix = output_vector_subset.reshape(pixels_side, pixels_side)

    return output_matrix

