import torch

from quantum_circuit import circuit

# Define a cost function that compares the circuit output to the target image
def cost_fn(params, original_image_angles, target_image_angles, nr_layers):
    cost = 0
    circuit_probs = circuit(params=params,
                            image_angles=original_image_angles,
                            nr_layers=nr_layers)  # returns 16 elements (basis probs)

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


# def cost_function(gen_image, target):
#     """
#     Calculates MSE between generated image and target image
#     """
#     loss = np.sum((gen_image - target) ** 2)
#     return loss


