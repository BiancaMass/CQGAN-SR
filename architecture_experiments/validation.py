import os
import torch
import matplotlib.pyplot as plt

from architecture_experiments import config
from quantum_circuit import circuit
from architecture_experiments.utils.image_processing import from_probs_to_image
from cost_function import cost_fn


def validate_model(trained_weights, validation_inputs, validation_targets):

    if not os.path.exists(config.OUTPUT_DIR_VALIDATE):
        os.makedirs(config.OUTPUT_DIR_VALIDATE)

    validation_losses = []

    for i in range(len(validation_inputs)):
        validation_input = validation_inputs[i]
        validation_target = validation_targets[i]

        probs_validation = circuit(params=trained_weights,
                                   flat_input_image=validation_input.flatten(),
                                   nr_layers=config.N_LAYERS,
                                   destination_qubit_indexes=config.DEST_QUBIT_INDEXES)

        validation_loss = cost_fn(params=trained_weights,
                                  original_image_flat=validation_input.flatten(),
                                  target_image_flat=validation_target.flatten(),
                                  nr_layers=config.N_LAYERS,
                                  dest_qubit_indexes=config.DEST_QUBIT_INDEXES)

        validation_output_image = from_probs_to_image(probs_validation)

        # Convert the tensor to a Python number if it's a tensor
        if isinstance(validation_loss, torch.Tensor):
            validation_losses.append(validation_loss.item())  # Use .item() to get a Python number
        else:
            validation_losses.append(
                validation_loss)  # If validation_loss is already a number

        # Plot and save validation images for each tuple
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(validation_input, cmap='gray', vmin=0, vmax=1)
        axs[0].set_title("Input Image")
        axs[0].axis('off')

        axs[1].imshow(validation_target, cmap='gray', vmin=0, vmax=1)
        axs[1].set_title("Target Image")
        axs[1].axis('off')

        axs[2].imshow(validation_output_image, cmap='gray', vmin=0, vmax=1)
        axs[2].set_title("Output Image")
        axs[2].axis('off')

        plt.suptitle(f"Validation Image Set {i +1} out of {len(validation_inputs)}")

        # Save the figure
        image_title = "validation_set_{}.png".format(i)
        image_path = os.path.join(config.OUTPUT_DIR_VALIDATE, image_title)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Print and plot validation loss and images for each pair
        print("\nValidation loss for image {}: {:.4f}".format(i, validation_loss))

    # Plot and save the validation loss graph
    plt.figure()
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, marker='o')
    plt.xlabel('Validation Set')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Over Multiple Sets')
    plt.grid(True)

    plot_title = "validation_losses.png"
    plot_path = os.path.join(config.OUTPUT_DIR, plot_title)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    average_validation_loss = sum(validation_losses) / len(validation_losses)
    print("\nAverage Validation loss is {:.4f}".format(average_validation_loss))

    return average_validation_loss, validation_losses
