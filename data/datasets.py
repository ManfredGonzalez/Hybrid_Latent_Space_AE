from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torchvision
import torchvision.transforms as transforms

import glob
import cv2
import numpy as np
import os
import random

class PineappleDataset(Dataset):
    """
    A custom PyTorch Dataset for loading pineapple images from a directory structure,
    with optional training, validation, and test splits, as well as image augmentation.

    Args:
        test_txt (str, optional): Path to a text file containing test image base names (without extension).
        path (str): Root directory containing the image files.
        train (bool): Whether to load training data. Mutually exclusive with 'val'.
        val (bool): Whether to load validation data. Mutually exclusive with 'train'.
        train_ratio (float): Proportion of the remaining (non-test) images used for training.
        val_ratio (float): Proportion of the remaining (non-test) images used for validation.
        resize_img (int): Desired width and height for resized images.
        augment (bool): Whether to apply data augmentation during training.
        augment_ratio (int): Ratio to augment the dataset for training.
    """

    def __init__(self, path, split='train', test_txt='test_list.txt', 
                 train_ratio=0.8, val_ratio=0.2, test_ratio=0.1,
                 resize_img=256, augment=False, augment_ratio=2, seed=42):
        assert split in ['train', 'val', 'test'], "split parameter must be 'train', 'val', or 'test'"

        self.path = path
        self.split = split
        self.resize_shape = (resize_img, resize_img)
        self.augment = augment and (split == 'train') # Only augment training data
        self.seed = seed

        # Get all image file paths sorted alphabetically
        all_images = sorted(glob.glob(os.path.join(path, "*")))

        ## Map base filenames to full paths
        image_dict = {os.path.splitext(os.path.basename(img))[0]: img for img in all_images}
        all_ids = sorted(list(image_dict.keys()))
        # 1. Handle Test Set Logic
        if test_txt is not None and os.path.isfile(test_txt):
            # Read existing test IDs
            with open(test_txt, "r") as f:
                test_ids = set(line.strip() for line in f if line.strip())
        else:
            # Create a new test set and save it
            random.seed(seed)
            shuffled_ids = all_ids.copy()
            random.shuffle(shuffled_ids)
            
            test_size = int(test_ratio * len(shuffled_ids))
            test_ids = set(shuffled_ids[:test_size])

            if test_txt is not None:
                os.makedirs(os.path.dirname(test_txt) or '.', exist_ok=True)
                with open(test_txt, "w") as f:
                    for tid in sorted(list(test_ids)):
                        f.write(f"{tid}\n")

        # 2. Handle Train/Val Logic (from remaining images)
        remaining_ids = [id_ for id_ in all_ids if id_ not in test_ids]
        remaining_images = [image_dict[id_] for id_ in remaining_ids]

        # Reset seed right before shuffling so train and val instances shuffle identically
        random.seed(seed)
        random.shuffle(remaining_images)

        # Normalize ratios just in case they don't exactly sum up
        total_remaining_ratio = train_ratio + val_ratio
        train_end = int((train_ratio / total_remaining_ratio) * len(remaining_images))

        # 3. Assign images based on the requested split
        if split == 'train':
            self.images = remaining_images[:train_end]
            if self.augment:
                self.images = self.images * augment_ratio
        elif split == 'val':
            self.images = remaining_images[train_end:]
        elif split == 'test':
            self.images = [image_dict[id_] for id_ in test_ids if id_ in image_dict]
        

    def __len__(self):
        """
        Returns the number of images in the current split.
        """
        return len(self.images)

    def transform_image(self, image_path):
        """
        Loads and preprocesses an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Transformed image tensor in CHW format, normalized to [0, 1].
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # Convert to RGB
        image = cv2.resize(image, self.resize_shape)      # Resize image

        # Apply augmentations with probability
        if self.augment:
            if random.random() < 0.3:
                image = cv2.flip(image, 1)  # Horizontal flip

            if random.random() < 0.2:
                beta = random.uniform(-10, 10)  # Brightness shift
                image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                center = (self.resize_shape[1] // 2, self.resize_shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1)
                image = cv2.warpAffine(image, M, self.resize_shape, borderMode=cv2.BORDER_REFLECT_101)

        # Normalize to [0, 1] and convert to CHW (Channels x Height x Width)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, idx):
        """
        Retrieves and transforms the image at the given index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            dict: Dictionary containing:
                - 'image': the preprocessed image tensor
                - 'idx': the index of the image
        """
        image = self.transform_image(self.images[idx])
        return {'image': image, 'idx': idx}
    
class TorchvisionDictWrapper(Dataset):
    """
    Wraps standard torchvision datasets to return a dictionary format
    expected by the existing training loop: {'image': tensor, 'idx': idx}
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # torchvision datasets return (image, label)
        image, label = self.dataset[idx]
        return {'image': image, 'idx': idx, 'label': label}

def get_benchmark_dataset(dataset_name, path='./datasets', split='train', val_ratio=0.2, resize_img=250, seed=42):
    """
    Downloads and prepares CIFAR10 or MNIST with Train/Val/Test splits.
    """
    # 1. Define standard transforms
    transform_list = []
    if resize_img != -1:
        transform_list.append(transforms.Resize((resize_img, resize_img)))
    
    transform_list.append(transforms.ToTensor())
    
    # If the model expects 3 channels, convert 1-channel MNIST to 3-channel
    if dataset_name.lower() == 'mnist':
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))
        
    transform = transforms.Compose(transform_list)

    # 2. Select the dataset class
    if dataset_name.lower() == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name.lower() == 'mnist':
        dataset_class = torchvision.datasets.MNIST
    else:
        raise ValueError(f"Unsupported benchmark dataset: {dataset_name}")

    # 3. Handle Test vs Train/Val splits
    is_train = split in ['train', 'val']
    
    # This automatically downloads if it doesn't exist in 'path'
    full_dataset = dataset_class(root=path, train=is_train, download=True, transform=transform)

    if split == 'test':
        return TorchvisionDictWrapper(full_dataset)

    # 4. Split the official Train set into Train and Val
    total_len = len(full_dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len
    
    # Use generator for reproducible splits
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_len, val_len], generator=generator)
    
    if split == 'train':
        return TorchvisionDictWrapper(train_subset)
    elif split == 'val':
        return TorchvisionDictWrapper(val_subset)