import os
import argparse
import numpy as np
import torch
# import torch_directml #directml does not support complex data types
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image

import config
from utils.image_processing import denorm
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

    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # Initialize an Adam optimizer for the generator.
    optimizer_G = Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    # Initialize an Adam optimizer for the critic.
    optimizer_C = Adam(critic.parameters(), lr=lr_D, betas=(b1, b2))

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
        for i, (input_images, target_images) in enumerate(dataloader):
            # Generate and save initial samples if not done already. # TODO: revisit
            # if not saved_initial:
            #     fixed_images = generator(input_image_flat)
            #     save_image(denorm(fixed_images),
            #                os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
            #     save_image(denorm(input_images), os.path.join(out_dir, 'real_samples.png'), nrow=5)
            #     saved_initial = True

            # Move real and target images to the specified device  (CPU or GPU).
            input_images = input_images.to(device)
            target_images = target_images.to(device)
            # Initialize the critic's optimizer (pytorch zero_grad).
            optimizer_C.zero_grad()

            # Give generator input image z to generate images
            fake_images = generator(input_images)

            # Compute the critic's predictions for real and fake images
            #  Returns a tensor with the result of discriminator (one number between 0 and 1),
            #  one for each image in the batch.
            real_validity = critic(target_images)  # Real (target) images
            fake_validity = critic(fake_images)  # Fake images.
            # Calculate the gradient penalty and adversarial loss.
            # TODO: delve in this function
            gradient_penalty = compute_gradient_penalty(critic, target_images, fake_images, device)
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
                fake_images = generator(input_images)
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
                    fixed_images = generator(input_images)
                    fixed_images = fixed_images.unsqueeze(1) # adds one dimension so that
                    # save_image works (1 b/c 1 color channel, as it is grayscale)
                    # TODO: should I denorm?
                    save_image(denorm(fixed_images),
                               os.path.join(out_dir, '{}.png'.format(batches_done)))
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