import config_a02
from training import train_model
from validation import validate_model
from image_processing import train_test_image_generator, from_probs_to_image
from my_utils import create_dataset

"""
Main training loop. Orchestrates the flow by calling functions from other scripts.
Consider creating command-line interface handling if needed.
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


images_per_training_set = config_a02.TRAINING_IMAGES_NUM
images_per_validation_set = config_a02.VALIDATION_IMAGES_NUMBER

train_images_inputs, train_images_targets = create_dataset(images_per_training_set)
validation_images_inputs, validation_images_targets = create_dataset(images_per_validation_set)


if __name__ == "__main__":
    main(training_inputs=train_images_inputs,
         training_targets=train_images_targets,
         val_inputs=validation_images_inputs,
         val_targets=validation_images_targets,
         n_layers=config_a02.N_LAYERS)
