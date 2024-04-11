# Wasserstein GAN loss function and auxiliary classifier loss function

import torch

def generator_loss(D, fake_images, labels, cond_weight=1.0):
    # Adversarial loss
    fake_scores, fake_labels = D(fake_images)
    loss = -torch.mean(fake_scores)

    # Conditional loss (if auxiliary classifier is used)
    if D.output_labels:
        criterion = torch.nn.CrossEntropyLoss()
        label_penalty_fakes = criterion(fake_labels, labels)
        loss += label_penalty_fakes * cond_weight

    return loss


def discriminator_loss(D, real_images, fake_images, labels, device, 
                       wgan_lambda=10.0, wgan_epsilon=0.001, wgan_target=1.0, cond_weight=1.0):
    # Real and fake scores
    real_scores, real_labels = D(real_images)
    fake_scores, fake_labels = D(fake_images)

    # Adversarial loss
    loss = torch.mean(fake_scores) - torch.mean(real_scores)

    # Gradient penalty
    alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    mixed_images = alpha * real_images + (1 - alpha) * fake_images
    mixed_images.requires_grad_(True)
    mixed_scores, _ = D(mixed_images)

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

    # Conditional loss (if auxiliary classifier is used)
    if D.output_labels:
        criterion = torch.nn.CrossEntropyLoss()
        label_penalty_reals = criterion(real_labels, labels)
        label_penalty_fakes = criterion(fake_labels, labels)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight

    return loss
