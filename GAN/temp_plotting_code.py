import matplotlib.pyplot as plt
import torch


# DELETE THIS TO NOT OVERWRITE
##################################################
input_images = torch.rand((8, 2, 2))
target_images = torch.rand((8, 4, 4))
fake_images = torch.rand((8, 4, 4))
##################################################

# PLOT INPUT AND TARGET IMAGES IN A GRID

# Create a figure with a grid layout
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))

# Iterate over each row and place input and output images next to each other
for i in range(8):
    # Calculate row and column index for the subplot
    row = i // 2
    col = (i % 2) * 2  # Multiply by 2 because each input has one output

    # Input image
    axs[row, col].imshow(input_images[i], cmap='gray', interpolation='none')
    axs[row, col].axis('off')  # Hide axes

    # Output image
    axs[row, col + 1].imshow(target_images[i], cmap='gray', interpolation='none')
    axs[row, col + 1].axis('off')  # Hide axes

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


# OUTPUT

# Create a figure with a 4x2 grid
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(4, 8))

# Plot the images in a 4x2 grid
for i in range(8):
    # Calculate row and column index for the subplot
    row = i // 2
    col = i % 2

    # Plot the image
    axs[row, col].imshow(fake_images.detach().numpy()[i], cmap='gray', interpolation='none')
    axs[row, col].axis('off')  # Hide axes

# Adjust layout
plt.tight_layout()
plt.show()

# for the normalized tensor
# Plotting the tensor in a 4x2 grid
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    # Select the tensor slice to plot
    tensor_slice = normalized[i][0].detach().numpy()
    img = ax.imshow(tensor_slice, cmap='gray')
    ax.axis('off')  # Turn off axis
# Add a colorbar
fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.6)
plt.show()
