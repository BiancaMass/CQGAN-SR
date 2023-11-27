# Consider making this into JSON instead, and/or changing some of these parameters into command
# line arguments (with argparse)
from datetime import datetime

from my_utils import destination_qubit_index_calculator

# Utils parameters
current_time = datetime.now()
OUTPUT_DIR = current_time.strftime("./output/%Y-%m-%d-%H%M")
OUTPUT_DIR_TRAIN = current_time.strftime("./output/%Y-%m-%d-%H%M/train")
OUTPUT_DIR_VALIDATE = current_time.strftime("./output/%Y-%m-%d-%H%M/valid")

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
TRAINING_IMAGES_NUM = 50
VALIDATION_IMAGES_NUMBER = 10
