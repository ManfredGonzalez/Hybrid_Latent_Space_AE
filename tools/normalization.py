import torch


def denormalize(tensor, dataset_name, device):
    """Reverses Z-score normalization for visualization and metrics."""
    if dataset_name.lower() == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean
    if dataset_name.lower() == 'imagenette':
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean
    # Pineapple and MNIST are already in [0, 1] range via Min-Max scaling
    return tensor
