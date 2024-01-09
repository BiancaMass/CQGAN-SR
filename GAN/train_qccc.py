import os
import argparse
import numpy as np
import torch
# import torch_directml #directml does not support complex data types
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image

import config
from utils.dataset import GeneratedImageDataset
from utils.wgan import compute_gradient_penalty
from GAN.QGCC import PQWGAN_QGCC


def train(layers, n_data_qubits, img_size, dest_qubit_indexes, batch_size, checkpoint):
    """
    Train the PQWGAN (generator and discriminator)
    Args:
        - layers (int): The number of layers in the generator.
        - n_data_qubits (int): The number of qubits, excluding the ancilla? Ancilla by default = 1.
        - batch_size (int): The batch size for training.
        - checkpoint (bool): Whether to create a checkpoint to resume training.

    Returns: None
    """
    device = torch.device("cpu")  # NOTE: GPU
    # device = torch_directml.device() # directML does not support complex data types
    n_epochs = 2  # NOTE: EPOCHS change back to 50

    # TODO: dataset structure is probably not ideal
    dataset = GeneratedImageDataset(100)

    ancillas = 0  # NOTE: change this if you want ancillas
    qubits = n_data_qubits + ancillas

    lr_D = 0.0002
    lr_G = 0.01
    b1 = 0
    b2 = 0.9
    lambda_gp = 10
    # How often to train gen and critic.E.g., if n_critic=5, train the gen every 5 critics.
    n_critic = 5
    sample_interval = 10
    # Default output folder name. Change if you want to include more params.
    out_dir = config.OUTPUT_DIR_TRAIN

    os.makedirs(out_dir, exist_ok=False)  # if dir already exists, raises an error

    gan = PQWGAN_QGCC(input_dimensions=(config.INPUT_ROWS, config.INPUT_COLS),
                      output_dimensions=(config.OUTPUT_ROWS, config.OUTPUT_COLS),
                      n_qubits=qubits,
                      n_ancillas=ancillas,
                      n_layers=layers,
                      dest_qubit_indexes=dest_qubit_indexes)

    # Assign the critic and generator models to the target device (e.g., GPU, CPU).
    critic = gan.critic.to(device)
    generator = gan.generator.to(device)

    # DataLoader from Pytorch to efficiently load and iterate over batches from the given dataset.
    # TODO: checks that this loads properly
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # Initialize an Adam optimizer for the generator.
    optimizer_G = Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    # Initialize an Adam optimizer for the critic.
    optimizer_C = Adam(critic.parameters(), lr=lr_D, betas=(b1, b2))

    # TODO: integrate Wasserstein distance or other cost function
    wasserstein_distance_history = []  # Store the Wasserstein distances
    saved_initial = False
    batches_done = 0

    # Load model checkpoints and training history if resuming training from a specific checkpoint.
    if checkpoint != 0: # TODO: see what happens when checkpoint is True
        critic.load_state_dict(torch.load(out_dir + f"/critic-{checkpoint}.pt"))
        generator.load_state_dict(torch.load(out_dir + f"/generator-{checkpoint}.pt"))
        wasserstein_distance_history = list(np.load(out_dir + "/wasserstein_distance.npy"))
        saved_initial = True
        batches_done = checkpoint

    # Begin training process of Generator and Discriminator.
    for epoch in range(n_epochs):
        print(f'Epoch number {epoch} \n')
        # Iterate over batches in the data loader. Goes over a batch of real and target images (_)
        for i, (real_images, _) in enumerate(dataloader):
            # Generate and save initial samples if not done already. # TODO: revisit
            # if not saved_initial:
            #     fixed_images = generator(input_image_flat)
            #     save_image(denorm(fixed_images),
            #                os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
            #     save_image(denorm(real_images), os.path.join(out_dir, 'real_samples.png'), nrow=5)
            #     saved_initial = True

            # Move real images to the specified device.
            real_images = real_images.to(device)
            # Initialize the critic's optimizer (pytorch zero_grad).
            optimizer_C.zero_grad()

            input_image_flat = real_images.flatten()  # this is a whole batch tho

            # Give generator input image z to generate images
            fake_images = generator(real_images)

            # Compute the critic's predictions for real and fake images.
            real_validity = critic(real_images)  # Real images. # TODO: change to 'label' image
            fake_validity = critic(fake_images)  # Fake images.
            # Calculate the gradient penalty and adversarial loss.
            gradient_penalty = compute_gradient_penalty(critic, real_images, fake_images, device)
            d_loss = -torch.mean(real_validity) + torch.mean(
                fake_validity) + lambda_gp * gradient_penalty
            # Calculate Wasserstein distance.
            wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)
            # Add distance for this batch.
            wasserstein_distance_history.append(wasserstein_distance.item())

            # Backpropagate and update the critic's weights.
            d_loss.backward()
            optimizer_C.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_images = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = critic(fake_images)
                g_loss = -torch.mean(fake_validity)

                # Backpropagate and update the generator's weights
                g_loss.backward()
                optimizer_G.step()

                # Print and log the training progress
                print(
                    f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Wasserstein Distance: {wasserstein_distance.item()}]")
                # Save Wasserstein distance history to a file
                np.save(os.path.join(out_dir, 'wasserstein_distance.npy'),
                        wasserstein_distance_history)
                # Update the total number of batches done
                batches_done += n_critic

                # Save generated images and model states at regular intervals
                if batches_done % sample_interval == 0:
                    fixed_images = generator(fixed_z)
                    save_image(denorm(fixed_images),
                               os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
                    torch.save(critic.state_dict(),
                               os.path.join(out_dir, 'critic-{}.pt'.format(batches_done)))
                    torch.save(generator.state_dict(),
                               os.path.join(out_dir, 'generator-{}.pt'.format(batches_done)))
                    print("saved images and state")


# Define the command-line arguments using argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layers", help="layers per sub-generators", type=int)
    parser.add_argument("-q", "--qubits", help="number of data qubits per sub-generator", type=int,
                        default=None)
    parser.add_argument("-b", "--batch_size", help="batch_size", type=int)
    parser.add_argument("-o", "--out_folder", help="output directory", type=str)
    parser.add_argument("-c", "--checkpoint", help="checkpoint to load from", type=int, default=0)
    # Parse the command-line arguments
    args = parser.parse_args()
    # Call the training function with the parsed arguments
    train(args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint)