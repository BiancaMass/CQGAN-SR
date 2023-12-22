from architecture_experiments.utils.image_processing import train_test_image_generator
from architecture_experiments import config


def destination_qubit_index_calculator(original_rows_num, original_cols_num, scaling_factor):
    """
    Computes destination qubit indexes for scaled image dimensions. Each pixel becomes a s*s
    square where s is the scaling factor. The initial pixel is located in the top left corner of
    the square. Future implementations might consider moving the pixel in the middle, which works
    especially well for odd numbers of s.

    Args:
        original_rows_num (int): Number of rows in the original image.
        original_cols_num (int): Number of columns in the original image.
        scaling_factor (int): Factor by which the image is scaled (same for both dims).

    Returns:
        List[int]: List of destination qubit indexes after scaling.

        """
    destination_cols = original_cols_num * scaling_factor
    destination_qubit_indexes = []
    for r in range(original_rows_num):
        for c in range(original_cols_num):
            destination_i, destination_j = r * scaling_factor, c * scaling_factor
            destination_index = destination_i * destination_cols + destination_j
            destination_qubit_indexes.append(destination_index)

    return destination_qubit_indexes


def create_dataset(images_per_set):
    images_inputs = []
    images_targets = []

    for i in range(images_per_set):
        image_LR, image_HR = train_test_image_generator(input_rows=config.INPUT_ROWS,
                                                        input_cols=config.INPUT_COLS,
                                                        scaling_factor=config.SCALING_FACTOR)
        images_inputs.append(image_LR)
        images_targets.append(image_HR)

    return images_inputs, images_targets


def save_variables(module, filename="variables.txt"):
    with open(filename, "w") as file:
        for key in dir(module):
            if not key.startswith("__"):
                value = getattr(module, key)
                file.write(f"{key}: {value}\n")
