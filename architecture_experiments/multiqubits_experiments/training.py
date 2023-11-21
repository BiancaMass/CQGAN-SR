import pennylane as qml
import numpy as np
# from pennylane import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from torch.autograd import Variable
from pennylane.optimize import GradientDescentOptimizer

from my_utils import cost_fn
from quantum_circuit import circuit
from image_processing import from_probs_to_image


def train_model(original_images, target_images, nr_qubits, nr_layers):

    # Convert image to a flattened array of rotation angles
    """From paper: the prior is restricted in the range [0,1) instead of [-pi, pi), to avoid too
    large of a search space for the generator, which would lead to poorer training """
    # Hence, comment out the pi conversion. Also, with current range 0,1 this brings it to 0, pi
    # image_angles = np.pi * original_img.flatten() # TODO: check this conversion is good
    # target_image_angles = np.pi * target_image.flatten()

    ### Initialize weights ###
    # TODO: using weights like paper [0,1)
    # scale_factor = 0.01  # Small scale factor
    # Each Rot gate needs 3 parameters, hence we have 3 random values per qubit per layer
    weights = np.random.rand(nr_layers, nr_qubits, 3)
    # weights = scale_factor * weights # uncomment if you want scale factor
    weights = Variable(torch.tensor(weights), requires_grad=True)

    ### Initialize the quantum circuit ###
    dev = qml.device("default.qubit", wires=nr_qubits)

    # set up the optimizer
    opt = torch.optim.Adam([weights], lr=0.01)  # lr from paper

    # number of steps in the optimization routine
    steps = 70  # TODO: hard-coded, change
    cost_vector = []
    ## optimization begins
    for i, (original_img, target_image) in enumerate(zip(original_images, target_images)):
        print('\n')
        print(f"Training on image pair {i + 1} of {len(original_images)}")
        # image_expanded = np.repeat(np.repeat(original_image, 2, axis=0), 2, axis=1)  # upscale directly
        original_flattened = original_img.flatten()
        target_flattened = target_image.flatten()

        # the final stage of optimization isn't always the best, so we keep track of
        # the best parameters along the way
        best_cost = cost_fn(params=weights,
                            original_image_angles=original_flattened,
                            target_image_angles=target_flattened,
                            nr_layers=nr_layers)
        best_weights = np.zeros((nr_layers, nr_qubits, 3))  # to store the best weights

        print(f"Cost after 0 steps is {best_cost}")

        for n in range(steps):
            opt.zero_grad()
            loss = cost_fn(params=weights,
                           original_image_angles=original_flattened,
                           target_image_angles=target_flattened,
                           nr_layers=nr_layers)
            cost_vector.append(float(loss))
            loss.backward()  # computes the gradient of the loss fun wrt weights
            opt.step()  # updates weights according to the computed gradient

            # keeps track of best parameters
            if loss < best_cost:
                best_cost = loss
                best_weights = weights

            # # Uncomment to save outputs every 10 iterations
            # output_dir = "./output/2023-11-10-1232"
            #
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)

            # Keep track of progress every 10 steps
            if n % 10 == 9 or n == steps - 1:
                print("Cost after {} steps is {:.4f}".format(n + 1, loss))
                # print(f"Weights after {n+1} is {weights[0][:2]}")  # to check if they are updating
            #
            #     # Uncomment to save outputs every 10 iterations
            #     current_probs = circuit(params=weights,
            #                             image_angles=original_flattened,
            #                             nr_layers=nr_layers)
            #     current_image_output = from_probs_to_image(current_probs)
            #     # Save the current image output with the iteration number as the title
            #     image_title = "iteration_{}.png".format(n)
            #     image_path = os.path.join(output_dir, image_title)
            #     fig, ax = plt.subplots()
            #     ax.imshow(current_image_output, cmap='gray')  # Use appropriate colormap if needed
            #     plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            #     plt.close(fig)

    final_probs = circuit(params=best_weights,
                          image_angles=original_flattened,
                          nr_layers=nr_layers)

    # computes the inverse cosine (arc cosine) of the square root of each prob
    # final_angles = torch.acos(torch.sqrt(final_probs)) # TODO: 65536 length

    # TODO: use this if you have ancilla
    # probs_given_ancilla_0 = final_probs[:2 ** (nr_qubits - 1)]  # Modify if you have ancillas
    # Normalize the probabilities
    # normalized_probs = probs_given_ancilla_0 / probs_given_ancilla_0.sum()

    normalized_probs = final_probs / final_probs.sum()

    # Rescale the probabilities to the range [-1, 1]
    max_val = torch.max(normalized_probs)
    final_post_processed_probs = ((normalized_probs / max_val) - 0.5) * 2

    plt.plot(cost_vector, label='Cost')
    # Add a title and labels
    plt.title('Cost Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

    return best_weights, final_post_processed_probs
