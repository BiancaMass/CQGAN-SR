import torch
import torch.autograd as autograd


def compute_gradient_penalty(critic, target_samples, fake_samples, device):
    """
    Computes the gradient penalty for enforcing the Lipschitz constraint in Wasserstein GANs.

    The G.P. is part of the loss function for the critic in WGAN-GP. It penalizes the model if the
    gradients of the critic with respect to the inputs do not have a norm of 1, enforcing the
    1-Lipschitz continuity condition required for a Wasserstein GAN.

    The function randomly interpolates the real and fake samples using epsilon-weighted average.
    It then computes the critic's scores for the interpolated samples.
    The gradient penalty is calculated by first computing the gradients of the critic scores
    with respect to the interpolated images. These gradients are then normalized and the
    deviation from the unit norm is squared and averaged over the batch to obtain the penalty.

    Args:
        critic (torch.nn.Module): The critic network.
        target_samples (torch.Tensor): The real samples from the data distribution.
            Shape: (batch_size, W, H)
        fake_samples (torch.Tensor): The fake samples generated by the generator.
            Shape: (batch_size, W, H)
        device (torch.device): The device on which the computations will be performed.
    Returns:
        torch.Tensor: The computed gradient penalty loss.
    """
    # Gradient penalty term which enforces the Lipschitz constraint (page 3 of paper)
    batch_size, W, H = target_samples.shape
    # The epsilon tensor, a weight for the interpolation (combination)
    # Ensures that each interpolated image is somewhere on the line between a real and a fake image.
    epsilon = torch.rand(batch_size, 1, 1).repeat(1, W, H).to(device)
    # Creates interpolated images by taking a convex combination of real images and generated images
    interpolated_images = (epsilon * target_samples + ((1 - epsilon) * fake_samples))
    # Calculate score of interpolated images by passing them thought the discriminator.
    # The critic estimates the Wasserstein distance between the distribution of real images and
    # the distribution of generated images. # Check: how does it do so?
    interpolated_scores = critic(interpolated_images)

    # Compute the gradients of the interpolated_scores w.r.t. the interpolated_images.
    gradients = autograd.grad(
        inputs=interpolated_images,
        outputs=interpolated_scores,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    # Compute the gradient penalty term.
    gradient_penalty = torch.mean((1. - torch.sqrt(1e-8 + torch.sum(gradients ** 2, dim=1))) ** 2)

    return gradient_penalty
