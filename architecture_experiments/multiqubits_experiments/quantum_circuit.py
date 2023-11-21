import pennylane as qml
import numpy as np
from image_processing import image_generator
import matplotlib.pyplot as plt


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

    for k in range(nr_qubits - 1):
        # Apply CNOT gates between adjacent qubits (i.e., linearly)
        qml.CNOT(wires=[k, k + 1])


# Define the quantum device
dev = qml.device("default.qubit", wires=16)  # TODO: hard-coded


@qml.qnode(dev, interface="torch")  # pytorch as interface for the training and backpropagation
def circuit(params, image_angles, nr_layers):
    dev.reset()
    nr_qubits = dev.num_wires
    # Check if the number of qubits is a multiple of the number of image angles
    if nr_qubits % len(image_angles) != 0:  # TODO: change if insert ancilla
        raise ValueError("Number of qubits must be a multiple of the number of image angles.")

    # Determine the number of qubits per angle group
    qubits_per_group = nr_qubits // len(image_angles)

    # Encode the image onto the quantum state using rotation gates
    # Apply the angles to the qubits in groups
    for i, angle in enumerate(image_angles):
        qubit_group = list(range(i * qubits_per_group, (i + 1) * qubits_per_group))
        for wire in qubit_group:
            qml.RY(angle, wires=wire)

    # repeatedly apply each layer in the circuit
    for j in range(nr_layers):
        layer(params=params, nr_qubits=nr_qubits, j=j)

    # Measure the state of the qubits and return the probabilities
    return qml.probs(wires=list(range(nr_qubits)))



