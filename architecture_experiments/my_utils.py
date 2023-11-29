from image_processing import train_test_image_generator
import config_a02


def destination_qubit_index_calculator(original_rows_num, original_cols_num):
    destination_cols = original_cols_num * 2 + 1
    destination_qubit_indexes = []
    for r in range(1, original_rows_num + 1):
        for c in range(1, original_cols_num + 1):
            original_i, original_j = r, c
            destination_i, destination_j = original_i * 2, original_j * 2
            destination_index = (destination_i - 1) * destination_cols + destination_j
            destination_qubit_indexes.append(destination_index)

    return destination_qubit_indexes


# import config_a02
def create_dataset(images_per_set):
    images_inputs = []
    images_targets = []

    for i in range(images_per_set):
        image_LR, image_HR = train_test_image_generator(input_rows=config_a02.INPUT_ROWS,
                                                        input_cols=config_a02.INPUT_COLS,
                                                        output_rows=config_a02.OUTPUT_ROWS,
                                                        output_cols=config_a02.OUTPUT_COLS)
        images_inputs.append(image_LR)
        images_targets.append(image_HR)

    return images_inputs, images_targets


def save_variables(module, filename="variables.txt"):
    with open(filename, "w") as file:
        for key in dir(module):
            if not key.startswith("__"):
                value = getattr(module, key)
                file.write(f"{key}: {value}\n")
