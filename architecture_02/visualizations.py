import matplotlib.pyplot as plt
import pennylane as qml


def plot_image(image, title=''):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def plot_images(images, titles):
    """
    Plots a series of images in a single figure with individual subplots.

    Parameters:
    - images (list): A list of images to be plotted.
    - titles (list): A list of titles for the subplots; should be the same length as images.

    Each image is displayed in its own subplot with the corresponding title.
    """
    # The number of images
    num_images = len(images)

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    # If there's only one image, axs is not a list so we make it a list for consistency
    if num_images == 1:
        axs = [axs]

    # Loop through each image and its corresponding title
    for ax, image, title in zip(axs, images, titles):
        ax.imshow(image, cmap='gray')
        ax.title.set_text(title)
        # ax.axis('off')  # Turn off axis

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()


# def circuit_drawer(circuit):
#     drawer = qml.draw(circuit)
#     print(drawer(img, weights))

# # Example usage
# print("Generated Image:")
# print(img)
# # Encode the image onto the quantum state
# circuit_output = encode_image_circuit(img, weights=weights)
# print("Circuit output:", circuit_output)
#
# # Visualize the circuit


