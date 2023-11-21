import numpy as np
import matplotlib.pyplot as plt


def create_image(rows, cols, random):
    if random:
        return np.random.rand(rows, cols)  # Generates random values between 0 and 1
    else:
        pixel_values = np.linspace(0, 1, rows).reshape(-1, 1)
        return np.tile(pixel_values, cols)


def image_plot(image1, image2):
    # Show the original images
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap="gray", interpolation="nearest")
    plt.title("Original Image 1")

    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap="gray", interpolation="nearest")
    plt.title("Original Image 2")

    plt.show()
