import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from visualizations import plot_image

np.random.seed(42)

# we generate a three-dimensional random vector by sampling
# each entry from a standard normal distribution
image = np.random.rand(2, 2)  # Example image
target_image = np.random.rand(4, 4)  # Example target image

plot_image(image, "original")
plot_image(target_image, "target")

# Convert image to a flattened array of rotation angles
image_angles = np.pi * image.flatten()
target_image_angles = np.pi * target_image.flatten()

# number of qubits in the circuit should match the number of pixels in the image
nr_qubits = len(image_angles)
nr_layers = 2

# randomly initialize parameters from a normal distribution
# 3 because Rot takes 3 params
params = np.random.normal(0, np.pi, (nr_layers, nr_qubits, 3)) # TODO should be a lot smaller
params = Variable(torch.tensor(params), requires_grad=True)


# a layer of the circuit ansatz
def layer(params, j):
    # Apply Rot gates for each layer of the circuit
    for i in range(nr_qubits):
        qml.Rot(params[j, i][0],
                params[j, i][1],
                params[j, i][2],
                wires=i)  # indexed: [layer, qubit, gate]

    for k in range(nr_qubits - 1):
        # Apply CNOT gates between adjacent qubits
        qml.CNOT(wires=[k, k + 1])


dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch")
def circuit(params):

    # Encode the image onto the quantum state using rotation gates
    for i, angle in enumerate(image_angles):
        qml.RY(angle, wires=i)

    # repeatedly apply each layer in the circuit
    for j in range(nr_layers):
        layer(params, j)

    # Measure the state of the qubits and return the probabilities
    return qml.probs(wires=list(range(nr_qubits)))


# Define a cost function that compares the circuit output to the target image
def cost_fn(params, target_image_angles):
    cost = 0
    circuit_probs = circuit(params)  # returns 16 elements (basis probs)

    # No need to convert to numpy, just use torch operations
    output_angles = torch.acos(torch.sqrt(circuit_probs))

    # Truncate the angles to only keep the first 4
    # Ensure you are truncating the tensor correctly, depending on the shape of circuit_probs
    truncated_output_angles = output_angles[:len(target_image_angles)]

    # Ensure target_image_angles is a tensor and has the correct shape
    # This operation should not require gradients
    target_image_angles_tensor = target_image_angles if isinstance(target_image_angles,
                                                                   torch.Tensor) else torch.tensor(
        target_image_angles, dtype=torch.float32)

    # Sum the squared differences between the output angles and the target angles
    # Make sure both tensors have the same shape for this operation
    cost += torch.sum((truncated_output_angles - target_image_angles_tensor) ** 2)

    return cost

# set up the optimizer
opt = torch.optim.Adam([params], lr=0.1)

# number of steps in the optimization routine
steps = 200

# the final stage of optimization isn't always the best, so we keep track of
# the best parameters along the way
best_cost = cost_fn(params=params, target_image_angles=target_image_angles)
best_params = np.zeros((nr_layers, nr_qubits, 3))

print(f"Cost after 0 steps is {best_cost}")

# optimization begins
for n in range(steps):
    opt.zero_grad()
    loss = cost_fn(params, target_image_angles)
    loss.backward()
    opt.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_cost = loss
        best_params = params

    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Cost after {} steps is {:.4f}".format(n + 1, loss))


final_probs = circuit(best_params)

final_angles = torch.acos(torch.sqrt(final_probs))


# Truncate the angles to only keep the first 4
# Ensure you are truncating the tensor correctly, depending on the shape of circuit_probs
truncated_final_angles = final_angles[:len(target_image_angles)]
truncated_final_angles = truncated_final_angles.detach()
plot_image(truncated_final_angles.reshape(4,4), "final")


# Create a figure with 3 subplots in 1 row
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust the figsize as needed

# Plot the original image
im0 = axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original")
fig.colorbar(im0, ax=axs[0])

# Plot the target image
im1 = axs[1].imshow(target_image, cmap='gray')
axs[1].set_title("Target")
fig.colorbar(im1, ax=axs[1])

# Plot the final image
# Ensure 'truncated_final_angles' is a NumPy array or a tensor converted to NumPy
# If it's a tensor, you might need to do 'truncated_final_angles.detach().numpy()'
im2 = axs[2].imshow(truncated_final_angles.reshape(4, 4), cmap='gray')
axs[2].set_title("Final")
fig.colorbar(im2, ax=axs[2])

# Set the layout of the subplots
plt.tight_layout()
plt.show()