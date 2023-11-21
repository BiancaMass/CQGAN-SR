import torch

from quantum_circuit import circuit


# Define a cost function that compares the circuit output to the target image
def cost_fn(params, original_image_flat, target_image_flat, nr_layers, dest_qubit_indexes):
    cost = 0
    circuit_probs = circuit(params=params,
                            flat_input_image=original_image_flat,
                            nr_layers=nr_layers,
                            destination_qubit_indexes=dest_qubit_indexes)

    post_measurement_probs = circuit_probs / torch.sum(circuit_probs)
    post_processed_patch = (post_measurement_probs / torch.max(post_measurement_probs)) # - 0.5) * 2
    truncated_output_tensor = post_processed_patch[:len(target_image_flat)]

    # output_angles = torch.acos(torch.sqrt(circuit_probs))

    # Truncate the angles to only keep the first 4
    # Ensure you are truncating the tensor correctly, depending on the shape of circuit_probs
    # truncated_output_angles = output_angles[:len(target_image_flat)]

    # Ensure target_image_flat is a tensor and has the correct shape
    # This operation should not require gradients
    target_image_tensor = target_image_flat if isinstance(target_image_flat,
                                                                 torch.Tensor) else torch.tensor(
        target_image_flat, dtype=torch.float32)

    # Sum the squared differences between the output angles and the target angles
    # Make sure both tensors have the same shape for this operation
    cost += torch.sum((truncated_output_tensor - target_image_tensor) ** 2)

    return cost

