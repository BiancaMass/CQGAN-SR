import os

import config

from utils.circuit_utils import destination_qubit_index_calculator

# Import the train function from your script
# Make sure to replace 'your_script' with the actual name of your script file
from train_qccc import train

def debug_train():
    # Set up debugging parameters
    layers = config.N_LAYERS  # number of layers in the generator
    n_data_qubits = config.N_QUBITS  # number of data qubits
    img_size = config.INPUT_ROWS * config.INPUT_COLS
    dest_qubit_indexes = destination_qubit_index_calculator(original_rows_num=config.INPUT_ROWS,
                                                            original_cols_num=config.INPUT_COLS,
                                                            scaling_factor=config.SCALING_FACTOR)
    batch_size = 8  # batch size for training
    n_epochs = config.N_EPOCHS
    # out_folder = config.OUTPUT_DIR  # output folder for saving results

    # Create the output directory if it doesn't exist
    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)

    checkpoint = 0  # Start training from the beginning

    # Call the train function with the debugging parameters
    train(layers, n_data_qubits, img_size, dest_qubit_indexes, batch_size, n_epochs, checkpoint)

if __name__ == "__main__":
    debug_train()
