import os

import config

# Import the train function from your script
# Make sure to replace 'your_script' with the actual name of your script file
from train_qccc import train

def debug_train():
    # Set up debugging parameters
    layers = config.N_LAYERS  # number of layers in the generator
    n_data_qubits = config.N_QUBITS  # number of data qubits
    batch_size = 8  # batch size for training
    out_folder = config.OUTPUT_DIR  # output folder for saving results

    # Create the output directory if it doesn't exist
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    checkpoint = 0  # Start training from the beginning

    # Call the train function with the debugging parameters
    train(layers, n_data_qubits, batch_size, out_folder, checkpoint)

if __name__ == "__main__":
    debug_train()
