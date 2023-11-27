import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from torch.autograd import Variable

import config_a02
from cost_function import cost_fn
from my_utils import destination_qubit_index_calculator
from quantum_circuit import circuit
from image_processing import from_probs_to_image


def train_model(original_images, target_images, nr_qubits, nr_layers):
    ### Initialize weights ###
    # Each Rot gate needs 3 parameters, hence we have 3 random values per qubit per layer
    weights = np.random.rand(nr_layers, nr_qubits, 3)  # sampled from a uniform  distr. over [0, 1).
    # convert into trainable param with torch framework
    weights = Variable(torch.tensor(weights), requires_grad=True)

    ### Initialize the quantum circuit ###
    dev = qml.device("default.qubit", wires=nr_qubits)
    destination_qubits_indexes_var = destination_qubit_index_calculator(config_a02.INPUT_ROWS,
                                                                        config_a02.INPUT_COLS)

    # set up the optimizer
    opt = torch.optim.Adam([weights], lr=0.01)  # lr from paper

    # number of steps in the optimization routine
    steps = config_a02.N_STEPS
    cost_vector = []

    # array to store the best weights
    best_weights = np.zeros((nr_layers, nr_qubits, 3))

    output_dir = config_a02.OUTPUT_DIR_TRAIN
    #
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # optimization begins
    for i, (original_img, target_image) in enumerate(zip(original_images, target_images)):
        print('\n')
        print(f"Training on image pair {i + 1} of {len(original_images)}")

        # Save current input-target image pair
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(original_img, cmap='gray')
        axs[0].set_title('Input image')
        axs[0].axis('off')  # Turn off axis

        axs[1].imshow(target_image, cmap='gray')
        axs[1].set_title('Target image')
        axs[1].axis('off')

        image_title = "{}_training_images.png".format(i)
        image_path = os.path.join(output_dir, image_title)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Flatten input and target images
        original_flattened = original_img.flatten()
        target_flattened = target_image.flatten()

        # The final stage of optimization isn't always the best, so keep track of best parameters
        best_cost = cost_fn(params=weights,
                            original_image_flat=original_flattened,
                            target_image_flat=target_flattened,
                            nr_layers=nr_layers,
                            dest_qubit_indexes=config_a02.DEST_QUBIT_INDEXES)

        print(f"Cost after 0 steps is {best_cost}")

        for n in range(steps):
            opt.zero_grad()  # Torch optimizer
            loss = cost_fn(params=weights,
                           original_image_flat=original_flattened,
                           target_image_flat=target_flattened,
                           nr_layers=nr_layers,
                           dest_qubit_indexes=config_a02.DEST_QUBIT_INDEXES)
            cost_vector.append(float(loss))
            loss.backward()  # computes the gradient of the loss function wrt weights
            opt.step()  # updates weights according to the computed gradient

            # keeps track of best parameters
            if loss < best_cost:
                best_cost = loss
                best_weights = weights

            # Keep track of progress every 15 steps
            if n % 15 == 14 or n == steps - 1:
                print("Cost after {} steps is {:.4f}".format(n + 1, loss))
                # print(f"Weights after {n+1} is {weights[0][:2]}")  # to check if they are updating

                # Uncomment to save outputs every 10 iterations
                current_probs = circuit(params=weights,
                                        flat_input_image=original_flattened,
                                        nr_layers=nr_layers,
                                        destination_qubit_indexes=destination_qubits_indexes_var)
                current_image_output = from_probs_to_image(current_probs)
                # Save the current image output with the iteration number as the title
                image_title = "{}_{}_iteration.png".format(i, n)
                image_path = os.path.join(output_dir, image_title)
                fig, ax = plt.subplots()
                ax.imshow(current_image_output, cmap='gray')  # Use appropriate colormap if needed
                plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

        # Current image pair results
        final_probs = circuit(params=best_weights,
                              flat_input_image=original_flattened,
                              nr_layers=nr_layers,
                              destination_qubit_indexes=destination_qubits_indexes_var)


        # # TODO: use this if you have ancilla
        # # probs_given_ancilla_0 = final_probs[:2 ** (nr_qubits - 1)]  # Modify if you have ancillas
        # # Normalize the probabilities
        # # normalized_probs = probs_given_ancilla_0 / probs_given_ancilla_0.sum()
        #
        # normalized_probs = final_probs / final_probs.sum()
        #
        # # Rescale the probabilities to the range [-1, 1]
        # max_val = torch.max(normalized_probs)
        # final_post_processed_probs = ((normalized_probs / max_val) - 0.5) * 2
        #
        # plt.plot(cost_vector, label='Cost')
        # # Add a title and labels
        # plt.title('Cost Over Iterations')
        # plt.xlabel('Iteration')
        # plt.ylabel('Cost')
        # # Add a legend
        # plt.legend()
        #
        # # Show the plot
        # plt.show()

    return best_weights
