import torch


def denormalize(tensor, dataset_name, device):
    """Reverses Z-score normalization for visualization and metrics."""
    if dataset_name.lower() == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean
    # imagenet uses the SAME (0.5, 0.5) normalization as imagenette (see
    # data.datasets.build_image_transform), so it must denormalize identically. Falling
    # through to the identity return below would leave images in [-1, 1] and silently
    # corrupt PSNR/SSIM/FID/KID and every logged reconstruction grid.
    if dataset_name.lower() in ('imagenette', 'imagenet'):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean
    # Pineapple and MNIST are already in [0, 1] range via Min-Max scaling
    return tensor
