# Consider making this into JSON instead, and/or changing some of these parameters into command
# line arguments (with argparse)
import os
from datetime import datetime

from my_utils import destination_qubit_index_calculator

# Image parameters
INPUT_ROWS = 2
INPUT_COLS = 1
OUTPUT_ROWS = INPUT_ROWS*2 + 1
OUTPUT_COLS = INPUT_COLS*2 + 1

# Circuit parameters
N_LAYERS = 3
N_QUBITS = OUTPUT_ROWS * OUTPUT_COLS
DEST_QUBIT_INDEXES = destination_qubit_index_calculator(INPUT_ROWS, INPUT_COLS)

# Training parameters
N_STEPS = 10
TRAINING_IMAGES_NUM = 40
VALIDATION_IMAGES_NUMBER = 12

# Utils parameters
ARCHITECTURE_NAME = "a-03"
current_time = datetime.now()
STRING_TIME = current_time.strftime("%Y-%m-%d-%H%M")
OUTPUT_DIR = os.path.join(f"./output/{ARCHITECTURE_NAME}/{STRING_TIME}_{N_LAYERS}-{N_STEPS}"
                          f"-{TRAINING_IMAGES_NUM}")
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, "train")
OUTPUT_DIR_VALIDATE = os.path.join(OUTPUT_DIR, "valid")
