import config_a02
from quantum_circuit import circuit
from training import train_model
from validation import validate_model
from image_processing import train_test_image_generator, from_probs_to_image
from visualizations import plot_images
from cost_function import cost_fn

"""
Main training loop. Orchestrates the flow by calling functions from other scripts.
Command-line interface handling if needed.
"""


def main(training_inputs, training_targets, val_inputs, val_targets, n_layers):
    # n_input_pixels = training_inputs[0].shape[0]*training_inputs[0].shape[1] # TODO: remove
    n_qubits = training_targets[0].shape[0] * training_targets[0].shape[1]

    # Train the model
    trained_weights = train_model(original_images=training_inputs,
                                  target_images=training_targets,
                                  nr_qubits=n_qubits,
                                  nr_layers=n_layers)

    average_validation_loss, validation_losses = validate_model(trained_weights=trained_weights,
                                                                validation_inputs=val_inputs,
                                                                validation_targets=val_targets)

    # # Call the circuit on another image # TODO: this should be done outside main()
    # validation_input, validation_target = train_test_image_generator(
    #                                             input_rows=config_a02.INPUT_ROWS,
    #                                             input_cols=config_a02.INPUT_COLS,
    #                                             output_rows=config_a02.OUTPUT_ROWS,
    #                                             output_cols=config_a02.OUTPUT_COLS)
    #
    # probs_validation = circuit(params=trained_weights,
    #                            flat_input_image=validation_input.flatten(),
    #                            nr_layers=n_layers,
    #                            destination_qubit_indexes=config_a02.DEST_QUBIT_INDEXES)
    #
    # validation_loss = cost_fn(params=trained_weights,
    #                           original_image_flat=validation_input.flatten(),
    #                           target_image_flat=validation_target.flatten(),
    #                           nr_layers=n_layers,
    #                           dest_qubit_indexes=config_a02.DEST_QUBIT_INDEXES)
    #
    # print("\n \n Validation loss is {:.4f}".format(validation_loss))
    #
    # validation_matrix = from_probs_to_image(probs_validation)
    #
    # plot_images([validation_input, validation_matrix],
    #             ["Test image", "Output Test image"])


images_per_training_set = config_a02.TRAINING_IMAGES_NUM
images_per_validation_set = config_a02.VALIDATION_IMAGES_NUMBER

# TODO: change this into a function
train_images_inputs = []
train_images_targets = []
validation_images_inputs = []
validation_images_targets = []

for i in range(images_per_training_set):
    image_LR, image_HR = train_test_image_generator(input_rows=config_a02.INPUT_ROWS,
                                                    input_cols=config_a02.INPUT_COLS,
                                                    output_rows=config_a02.OUTPUT_ROWS,
                                                    output_cols=config_a02.OUTPUT_COLS)
    train_images_inputs.append(image_LR)
    train_images_targets.append(image_HR)

for i in range(images_per_validation_set):
    image_LR, image_HR = train_test_image_generator(input_rows=config_a02.INPUT_ROWS,
                                                    input_cols=config_a02.INPUT_COLS,
                                                    output_rows=config_a02.OUTPUT_ROWS,
                                                    output_cols=config_a02.OUTPUT_COLS)
    validation_images_inputs.append(image_LR)
    validation_images_targets.append(image_HR)

if __name__ == "__main__":
    main(training_inputs=train_images_inputs,
         training_targets=train_images_targets,
         val_inputs=validation_images_inputs,
         val_targets=validation_images_targets,
         n_layers=config_a02.N_LAYERS)
