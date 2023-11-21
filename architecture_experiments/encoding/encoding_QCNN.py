import pennylane as qml
import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import torch

from image_operations import create_image, image_plot

""" This script performs angle encoding with RY gates of simple black and white images
Angle encoding requires as many wires (qubits) as pixels, and is therefore not very space efficient.
"""


###########################
#     Image generation    #
###########################

nrows = ncols = 5
nwires = nrows * ncols
# n_layers = 1

# Create two grayscale images
image1 = create_image(nrows, ncols, True)
image2 = create_image(nrows, ncols, False)

# Show the created images
image_plot(image1=image1, image2=image2)

################################
# Angle Encoding and Measuring #
################################

# Initialize a PennyLane default.qubit device, simulating a system of n qubits
dev = qml.device("default.qubit", wires=nwires)


# qnode representing a quantum circuit consisting of:
# 1. An embedding layer of local Ry rotations (with angles scaled by a factor of π);
# 2. (A random circuit of n_layers;)
# 3. A final measurement in the computational basis, estimating 4 expectation values.

# Random circuit parameters
# rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, nwires))

@qml.qnode(dev, interface="autograd")
def circuit(image):
    flat_image = image.flatten()
    # ENCODING of classical input values (STEP 1)
    for j in range(nwires):
        qml.RY(np.pi * flat_image[j], wires=j)

    # Random quantum circuit (STEP 2)
    # RandomLayers(rand_params, wires=list(range(nwires)))

    # Measurement producing 4 classical output values (STEP 3)
    return [qml.expval(qml.PauliZ(j)) for j in range(nwires)]




# Number of images and grid dimensions
num_images = 25
grid_size = int(np.ceil(np.sqrt(num_images)))

# Create a subplot grid
fig, axes = plt.subplots(grid_size, grid_size, figsize=(nrows, ncols))

for i in range(num_images):
    # Generate and process an image using the circuit function
    image = circuit(image2)  # Replace this with your actual image or image generation logic
    image = np.vstack(image)
    image = image.reshape(nrows, ncols)

    # Plot the image in the grid
    row = i // grid_size
    col = i % grid_size
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].axis('off')

# Adjust layout and show the plot
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing between subplots
plt.suptitle('Decoded Images')  # Overall title for the entire plot
plt.show()
