import pennylane as qml
import io
import sys
import os
from PIL import Image, ImageDraw

import config
from training import train_model
from validation import validate_model
from my_utils import create_dataset, save_variables
from quantum_circuit import circuit

"""
Main training loop. Calls training and validation functions.
Consider creating command-line interface handling if needed.
"""


def main(training_inputs, training_targets, val_inputs, val_targets, n_layers):
    n_qubits = training_targets[0].shape[0] * training_targets[0].shape[1]

    # TRAINING
    trained_weights = train_model(original_images=training_inputs,
                                  target_images=training_targets,
                                  nr_qubits=n_qubits,
                                  nr_layers=n_layers)

    # CREATE AN IMAGE OF THE CIRCUIT
    # Redirect print output to a buffer
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    image_for_drawing_circuit = flattened_lst = [item for sublist in training_inputs[0] for item in
                                                 sublist]
    print(qml.draw(circuit)(trained_weights, image_for_drawing_circuit, n_layers,
                            config.DEST_QUBIT_INDEXES))

    # Reset standard output and get buffer content
    sys.stdout = old_stdout
    print_output = buffer.getvalue()

    # Create an image from the print output
    image = Image.new('RGB', (700, 600), color=(255, 255, 255))  # Adjust size as needed
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), print_output, fill=(0, 0, 0))
    circuit_image_path = os.path.join(config.OUTPUT_DIR, 'circuit_diagram.png')
    image.save(circuit_image_path)

    # Save variables as a text file
    text_variables_path = os.path.join(config.OUTPUT_DIR, 'variables.text')
    save_variables(config, filename=text_variables_path)

    # VALIDATION
    average_validation_loss, validation_losses = validate_model(trained_weights=trained_weights,
                                                                validation_inputs=val_inputs,
                                                                validation_targets=val_targets)


images_per_training_set = config.TRAINING_IMAGES_NUM
images_per_validation_set = config.VALIDATION_IMAGES_NUMBER

train_images_inputs, train_images_targets = create_dataset(images_per_training_set)
validation_images_inputs, validation_images_targets = create_dataset(images_per_validation_set)


if __name__ == "__main__":
    main(training_inputs=train_images_inputs,
         training_targets=train_images_targets,
         val_inputs=validation_images_inputs,
         val_targets=validation_images_targets,
         n_layers=config.N_LAYERS)
