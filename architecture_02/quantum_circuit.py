import pennylane as qml

import config_a02


# a layer of the circuit ansatz
def layer(params, nr_qubits, j):
    """
    Apply a single layer of the quantum circuit ansatz.

    This function applies rotation gates (Rot) to each qubit in the layer, followed by
    a series of CNOT gates applied linearly.

    Parameters:
    - params (np.ndarray): A 3-dimensional array of shape (layers, qubits, parameters)
      containing the rotation angles for the Rot gates. The parameters for the Rot gates
      are 3, and correspond to the order of rotation about the X (θ - theta), Y (ϕ - phi),
      and Z (λ - lambda)  axes.
    - nr_qubits (int): The number of qubits in the circuit.
    - j (int): The index of the current layer for which the gates are being applied.

    Returns:
    None

    Note: This function does not return a value but modifies the quantum circuit's state
    by applying gates to it.
    """

    # Apply Rot gates for each layer of the circuit
    for i in range(nr_qubits):
        qml.Rot(params[j, i][0],
                params[j, i][1],
                params[j, i][2],
                wires=i)  # indexed: [layer, qubit, param for the specific gate (θ, ϕ, λ) ]

    # for k in range(nr_qubits - 1):
    #     # Apply CNOT gates between adjacent qubits (i.e., linearly)
    #     qml.CNOT(wires=[k, k + 1])


# Define the quantum device
dev = qml.device("default.qubit", wires=config_a02.N_QUBITS)


@qml.qnode(dev, interface="torch")  # pytorch as interface for the training and backpropagation
def circuit(params, flat_input_image, nr_layers: int, destination_qubit_indexes):
    """
    The quantum circuit for increasing image resolution.
    Parameters:

    - params (torch.Tensor): A tensor of parameters (requiring gradients) representing the angles
      for rotation gates in the quantum circuit.

    - flat_input_image (numpy.ndarray): The input image, flattened in a 1D numpy array.

    - nr_layers (int): The number of layers in the quantum circuit.

    - destination_qubit_indexes (list of int): Indices of the original pixels in the flattened
      output (upscaled) image. For example, in a 2x1 original image, this would be [5, 11]. This is
      used to assign the pixel value to be encoded with a RY gate to the correct qubit.

    """
    dev.reset()
    nr_qubits = dev.num_wires

    # Encode the original image onto the quantum state using rotation gates on the relevant qubits
    for pixel, qubit_index in enumerate(destination_qubit_indexes):
        qml.RY(flat_input_image[pixel], wires=qubit_index)

    # All the other are 0s # TODO: check, does Pennylane do this by default?

    # Apply CNOT between the original pixels
    # TODO: when more than 2, this will have to be transformed in a loop where only relevant (
    #  originally adjacent) pixels have a CNOT between them.
    qml.CNOT(wires=destination_qubit_indexes)

    # Apply Hadamard to all qubits
    for wire in range(nr_qubits):
        qml.Hadamard(wires=wire)

    # Parameterized layer - apply repeatedly
    for j in range(nr_layers):
        layer(params=params, nr_qubits=nr_qubits, j=j)

    # Measure the state of the qubits and return the probabilities
    return qml.probs(wires=list(range(nr_qubits)))
