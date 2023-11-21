import config_a02
from quantum_circuit import circuit
from training import train_model
from image_processing import train_test_image_generator, from_probs_to_image
from visualizations import plot_images
from cost_function import cost_fn

"""
Main training loop. Orchestrates the flow by calling functions from other scripts.
Command-line interface handling if needed.
"""


def main(training_images, target_images, n_layers):
    n_input_pixels = training_images[0].shape[0]*training_images[0].shape[1]
    n_qubits = target_images[0].shape[0]*target_images[0].shape[1]

    # Train the model
    trained_weights = train_model(original_images=training_images,
                                  target_images=target_images,
                                  nr_qubits=n_qubits,
                                  nr_layers=n_layers)

    # Call the circuit on another image # TODO: this should be done outside main()
    validation_input, validation_target = train_test_image_generator(
                                                input_rows=config_a02.INPUT_ROWS,
                                                input_cols=config_a02.INPUT_COLS,
                                                output_rows=config_a02.OUTPUT_ROWS,
                                                output_cols=config_a02.OUTPUT_COLS)

    probs_validation = circuit(params=trained_weights,
                               flat_input_image=validation_input.flatten(),
                               nr_layers=n_layers,
                               destination_qubit_indexes=config_a02.DEST_QUBIT_INDEXES)

    validation_loss = cost_fn(params=trained_weights,
                              original_image_flat=validation_input.flatten(),
                              target_image_flat=validation_target.flatten(),
                              nr_layers=n_layers,
                              dest_qubit_indexes=config_a02.DEST_QUBIT_INDEXES)

    print("\n \n Validation loss is {:.4f}".format(validation_loss))

    validation_matrix = from_probs_to_image(probs_validation)

    plot_images([validation_input, validation_matrix],
                ["Test image", "Output Test image"])


images_per_training_set = config_a02.TRAINING_IMAGES_NUM

train_images = []
validation_images = []

for i in range(images_per_training_set):
    image_LR, image_HR = train_test_image_generator(input_rows=config_a02.INPUT_ROWS,
                                                    input_cols=config_a02.INPUT_COLS,
                                                    output_rows=config_a02.OUTPUT_ROWS,
                                                    output_cols=config_a02.OUTPUT_COLS)
    train_images.append(image_LR)
    validation_images.append(image_HR)

if __name__ == "__main__":
    main(training_images=train_images,
         target_images=validation_images,
         n_layers=config_a02.N_LAYERS)
