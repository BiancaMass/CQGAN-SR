import numpy as np
import pennylane as qml
import torch

from quantum_circuit import circuit
from training import train_model
from image_processing import train_test_image_generator, from_probs_to_image
from visualizations import plot_image, plot_images

"""
Main training loop. Orchestrates the flow by calling functions from other scripts.
Command-line interface handling if needed.
"""


# TODO input will have to be a training set,not just one image

def main(training_images, target_images, n_layers):
    # Generate image
    n_pixels = training_images[0].shape[0]*training_images[0].shape[1]
    n_qubits = target_images[0].shape[0]*target_images[0].shape[1]

    if n_qubits != n_pixels * 4:  # TODO: scaling factor hard coded
        raise ValueError("The number of qubits does not match the expected size based on the "
                         "number of pixels and scaling factor.")

    # TODO: this was for when I only had 1 image to train
    # image_expanded = np.repeat(np.repeat(original_image, 2, axis=0), 2, axis=1)  # upscale directly
    #
    # print(f'Original image:{original_image}')
    # print(f'Original image, upscaled:{image_expanded}')
    # print(f'Target image:{target_image}')

    # Train the model
    trained_weights, post_processed_probs = train_model(original_images=training_images,
                                                        target_images=target_images,
                                                        nr_qubits=n_qubits,
                                                        nr_layers=n_layers)



    # TODO: this does not work cause the prob. output is in the shape 2^n_qubits-1 (1630)
    '''From the paper: Since the size of the outputs from the quantum circuit are power of 2s,
    we only keep the first HW/P pixels to create a patch with the correct dimensions,
    where I would say that P stands for number of patches'''
    # Keep only the first H*W pixels
    # probs_subset = post_processed_probs[:n_qubits]
    #
    # # Reshape the probabilities back into a 4x4 matrix to visualize them as an image
    # post_processed_matrix = probs_subset.reshape((n_pixels, n_pixels))

    # plot_images([original_image, image_expanded, target_image,
    #              post_processed_matrix.detach().numpy()],
    #             ["Original image", "Upscaled original", "target image", "Output image"])

    # Call the circuit on another image
    validation_input, validation_target = train_test_image_generator(original_px_per_side=2, scale_factor=2)
    probs_validation = circuit(params=trained_weights,
                               image_angles=validation_input.flatten(),
                               nr_layers=n_layers)

    # normalized_probs_validation = probs_validation / probs_validation.sum()
    # max_val_validation = torch.max(normalized_probs_validation)
    # final_probs2 = ((normalized_probs_validation / max_val_validation) - 0.5) * 2
    # probs_subset_validation = final_probs2[:n_qubits]
    # validation_matrix = probs_subset_validation.reshape((n_pixels, n_pixels))
    #
    validation_matrix = from_probs_to_image(probs_validation)

    plot_images([validation_input, validation_matrix.detach().numpy()],
                ["Test image", "Output Test image"])


# pixel values are either 0 (black) or 1 (white)
images_per_training_set = 10

train_images = []
validation_images = []

for i in range(images_per_training_set):
    image_LR, image_HR = train_test_image_generator(original_px_per_side=2, scale_factor=2)
    train_images.append(image_LR)
    validation_images.append(image_HR)

if __name__ == "__main__":
    main(training_images=train_images,
         target_images=validation_images,
         n_layers=5)
