import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class PQWGAN_CC():
    def __init__(self, image_size, n_qubits, n_ancillas, n_layers):
        self.image_shape = (image_size, image_size)
        self.critic = self.ClassicalCritic(self.image_shape)
        self.generator = self.QuantumGenerator(n_qubits, n_ancillas, n_layers, self.image_shape)

    class ClassicalCritic(nn.Module):
        # takes the image either real or generated and decides which one it is
        def __init__(self, image_shape):
            super().__init__()
            self.image_shape = image_shape

            self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.fc3(x)

    class QuantumGenerator(nn.Module):
        def __init__(self, n_qubits, n_ancillas, n_layers, image_shape):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_ancillas = n_ancillas
            self.n_layers = n_layers
            self.q_device = qml.device("default.qubit", wires=n_qubits)

            ### Initialize weights ###
            # Each Rot gate needs 3 parameters, hence we have 3 random values per qubit per layer
            # sampled from a uniform  distr. over [0, 1).
            weight = np.random.rand(self.n_layers, self.n_qubits, 3)
            # convert into trainable param with torch framework
            self.params = Variable(torch.tensor(weight), requires_grad=True)
            self.qnode = qml.QNode(self.circuit, self.q_device, interface="torch")

            self.image_shape = image_shape

        def forward(self, x):
            image_pixels = self.image_shape[2] ** 2  # ASSUMES square images

            output_images = torch.Tensor(x.size(0), 0)

            for input_image in x:
                generator_output = self.partial_trace_and_postprocess(
                                                            input_image,
                                                            self.params).float().unsqueeze(0)
                output_images = torch.cat((output_images, generator_output), 1)

            final_out = output_images.view(output_images.shape[0], *self.image_shape)

            return final_out

        def circuit(self, input_image, weights, destination_qubit_indexes):

            for pixel, qubit_index in enumerate(destination_qubit_indexes):
                qml.RY(input_image[pixel], wires=qubit_index)

            # Apply Hadamard to all qubits
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)

            for i in range(self.n_layers):
                for j in range(self.n_qubits):
                    # indexed: [layer, qubit, param for the specific gate (θ, ϕ, λ) ]
                    qml.Rot(weights[i, j][0],
                            weights[i, j][1],
                            weights[i, j][2],
                            wires=j)

            return qml.probs(wires=list(range(self.n_qubits)))

        def partial_trace_and_postprocess(self, input_image, weights):
            probs = self.qnode(input_image, weights)
            probs_given_ancilla_0 = probs[:2 ** (self.n_qubits - self.n_ancillas)]
            post_measurement_probs = probs_given_ancilla_0 / torch.sum(probs_given_ancilla_0)

            # normalize between 0, 1
            post_processed_patch = (post_measurement_probs / torch.max(post_measurement_probs))

            return post_processed_patch


if __name__ == "__main__":
    gen = PQWGAN_CC(image_size=16, n_qubits=5, n_ancillas=1, n_layers=1).generator
    print(qml.draw(gen.qnode)(torch.rand(5), torch.rand(1, 5, 3)))