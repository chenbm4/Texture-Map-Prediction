# Wasserstein GAN loss function and auxiliary classifier loss function

import torch

def generator_loss(D, fake_images):
    # Adversarial loss
    fake_scores = D(fake_images)
    loss = -torch.mean(fake_scores)
    return loss


def discriminator_loss(D, real_images, fake_images, device, 
                       wgan_lambda=10.0, wgan_epsilon=0.001, wgan_target=1.0):
    # Real and fake scores
    real_scores = D(real_images)
    fake_scores = D(fake_images)

    # Adversarial loss
    loss = torch.mean(fake_scores) - torch.mean(real_scores)

    # Gradient penalty
    alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    mixed_images = alpha * real_images + (1 - alpha) * fake_images
    mixed_images.requires_grad_(True)
    mixed_scores = D(mixed_images)

    gradients = torch.autograd.grad(
        outputs=torch.sum(mixed_scores), 
        inputs=mixed_images, 
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradient_norm - wgan_target) ** 2).mean()
    loss += gradient_penalty * wgan_lambda

    # Epsilon penalty
    epsilon_penalty = torch.mean(real_scores ** 2)
    loss += epsilon_penalty * wgan_epsilon

    return loss
