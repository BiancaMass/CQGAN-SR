# Consider making this into JSON instead, and/or changing some of these parameters into command
# line arguments (with argparse)

from my_utils import destination_qubit_index_calculator

# Image parameters
INPUT_ROWS = 2
INPUT_COLS = 1
OUTPUT_ROWS = INPUT_ROWS*2 + 1
OUTPUT_COLS = INPUT_COLS*2 + 1

# Circuit parameters
N_LAYERS = 5
N_QUBITS = OUTPUT_ROWS * OUTPUT_COLS
DEST_QUBIT_INDEXES = destination_qubit_index_calculator(INPUT_ROWS, INPUT_COLS)

# Training parameters
N_STEPS = 40
TRAINING_IMAGES_NUM = 20
