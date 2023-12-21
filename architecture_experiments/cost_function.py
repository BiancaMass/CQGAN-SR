import torch

from quantum_circuit import circuit


# TODO: change into better cost function
def cost_fn(params, original_image_flat, target_image_flat, nr_layers, dest_qubit_indexes):
    """
        Calculates the distance between the generated image and target image, i.e., the cost used
        to train the circuit. The function computes the quantum circuit probabilities, processes
        the output,  and calculates the mean squared error between the processed output and the
        target image.

        Args:
            params (Tensor): Parameters for the quantum circuit.
            original_image_flat: Flattened tensor of the original image.
            target_image_flat: Flattened tensor or array of the target image.
            nr_layers (int): Number of layers in the quantum circuit.
            dest_qubit_indexes (List[int]): List of destination qubit indices.

        Returns: The calculated cost based on the squared difference between the processed image
                 and the target image.
        """
    cost = 0
    circuit_probs = circuit(params=params,
                            flat_input_image=original_image_flat,
                            nr_layers=nr_layers,
                            destination_qubit_indexes=dest_qubit_indexes)

    post_measurement_probs = circuit_probs / torch.sum(circuit_probs)
    post_processed_patch = (post_measurement_probs / torch.max(post_measurement_probs))
    truncated_output_tensor = post_processed_patch[:len(target_image_flat)]

    # Ensure target_image_flat is a tensor and has the correct shape
    target_image_tensor = target_image_flat if isinstance(target_image_flat,
                                                          torch.Tensor) else torch.tensor(
        target_image_flat, dtype=torch.float32)

    # Sum the squared differences between the output pixels and the target pixels
    cost += torch.sum((truncated_output_tensor - target_image_tensor) ** 2)

    return cost
