from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torchvision
import torchvision.transforms as transforms

import glob
import cv2
import numpy as np
import os
import random
from PIL import Image

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

class FlatImageDataset(Dataset):
    """
    A generic PyTorch Dataset for a flat directory of images.
    Perfect for unsupervised/generative models like VAEs.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Filter and list only image files to avoid reading system artifacts
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.image_files = [
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(valid_extensions)
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
        
        if self.transform:
            image = self.transform(image)
            
        # For a VAE, we only care about the image itself. 
        # Returning a dummy label (0) keeps it compatible with generic training loops.
        label = 0
        
        # Match your repository's dictionary format exactly
        return {
            'image': image, 
            'idx': idx, 
            'label': label
        }

# Imagenette's 10 WordNet ids, in the canonical (sorted) order, with readable names.
# Used to label a FLATTENED imagenette directory, where the class is only recoverable from
# the filename prefix (n01440764_10026.JPEG -> n01440764 -> tench).
IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def find_labels_csv(root_dir, filename="noisy_imagenette.csv"):
    """Locate fastai's imagenette label CSV, which ships inside the imagenette2 tarball.

    Looked for in the split directory itself and then in its parent (where the tarball puts
    it, alongside train/ and val/). Returns None when absent.
    """
    for cand in (os.path.join(root_dir, filename),
                 os.path.join(os.path.dirname(os.path.abspath(root_dir)), filename)):
        if os.path.isfile(cand):
            return cand
    return None


def load_labels_csv(csv_path, label_column="noisy_labels_0"):
    """{basename: class_key} from fastai's imagenette CSV.

    Keyed by BASENAME, not by the CSV's `path` column, so it works no matter how the tree was
    flattened or re-nested. Basenames are unique across all 13394 imagenette images.

    label_column: 'noisy_labels_0' is the clean ground truth. The other columns
    (noisy_labels_1/5/25/50) are fastai's deliberately corrupted variants -- only use them if
    you are explicitly studying label noise.
    """
    import csv as _csv

    with open(csv_path, newline='') as f:
        reader = _csv.DictReader(f)
        if label_column not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} has no column {label_column!r}; found {reader.fieldnames}.")
        return {os.path.basename(r["path"]): r[label_column] for r in reader}


class LabeledImageDataset(Dataset):
    """Image dataset that carries a REAL class label, for class-conditional generation.

    FlatImageDataset returns a constant label 0, which is fine for autoencoder training but
    makes conditional generation vacuous. This one recovers the class from whichever label
    source is actually available, in order of trustworthiness:

      1. NESTED (torchvision ImageFolder layout): root/<class>/<image>. Class = subdirectory.
      2. LABEL CSV: fastai's noisy_imagenette.csv (auto-discovered in the split directory or
         its parent), keyed by image basename.
      3. FILENAME PREFIX: n01440764_10026.JPEG -> n01440764.

    The CSV outranks the filename prefix for a concrete reason: imagenette2 is NOT uniformly
    named. About 4% of the images in BOTH splits (366 of 9469 train, 134 of 3925 val) come
    from ImageNet's validation set and are named ILSVRC2012_val_00000293.JPEG, which carries
    no class at all. Parsing prefixes on a flattened tree would therefore invent a bogus
    'ILSVRC2012' class holding ~500 images of ten different real classes. The CSV covers every
    file exactly once, so it is used whenever it is present.

    `classes` can be passed explicitly to force the train and val splits onto the SAME label
    indices (a val split missing one class would otherwise silently shift every id).
    Raises if no label structure can be found -- silently falling back to a single class
    would turn conditional training into unconditional training without any warning.
    """

    def __init__(self, root_dir, transform=None, classes=None, class_names=None,
                 labels_csv=None, label_column="noisy_labels_0"):
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        entries = sorted(os.listdir(root_dir))
        subdirs = [d for d in entries if os.path.isdir(os.path.join(root_dir, d))]

        csv_path = labels_csv or find_labels_csv(root_dir)
        csv_labels = load_labels_csv(csv_path, label_column) if csv_path else None
        self.labels_csv = csv_path

        samples = []   # (path, class_key)
        if subdirs:
            for d in subdirs:
                for f in sorted(os.listdir(os.path.join(root_dir, d))):
                    if f.lower().endswith(_IMAGE_EXTENSIONS):
                        samples.append((os.path.join(root_dir, d, f), d))
            self.layout = "nested"
        else:
            for f in entries:
                if not f.lower().endswith(_IMAGE_EXTENSIONS):
                    continue
                if csv_labels is not None:
                    key = csv_labels.get(f)          # authoritative; None -> reported below
                else:
                    key = f.split('_')[0]
                    if key == os.path.splitext(f)[0]:
                        # No '_' in the name -> no class information in the filename.
                        key = None
                samples.append((os.path.join(root_dir, f), key))
            self.layout = "flat+csv" if csv_labels is not None else "flat"

        if not samples:
            raise RuntimeError(f"No images found under {root_dir}.")
        unlabeled = [os.path.basename(p) for p, key in samples if key is None]
        if unlabeled:
            if csv_labels is not None:
                raise RuntimeError(
                    f"{len(unlabeled)} of {len(samples)} images in {root_dir} are missing from "
                    f"{csv_path} (e.g. {unlabeled[:3]}). The label CSV must cover the whole "
                    f"split -- check it is the one that shipped with THIS copy of the dataset."
                )
            raise RuntimeError(
                f"Cannot recover class labels from {root_dir}: it is a flat directory whose "
                f"filenames carry no '<class>_<id>' prefix, it has no per-class subdirectories, "
                f"and no noisy_imagenette.csv was found in it or its parent. Class-conditional "
                f"training needs labels -- copy the CSV from the imagenette2-320 tarball next "
                f"to train/ and val/, or set labels_csv in the config."
            )

        if classes is None:
            classes = sorted({key for _, key in samples})
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        unknown = {key for _, key in samples if key not in self.class_to_idx}
        if unknown:
            hint = ""
            if "ILSVRC2012" in unknown:
                # ~4% of imagenette2's images (both splits) come from ImageNet's VALIDATION set
                # and are named ILSVRC2012_val_00000293.JPEG, which carries no class. Their
                # labels live in noisy_imagenette.csv (or in the per-wnid directory names).
                hint = (" 'ILSVRC2012' is not a real class -- those are the ~4% of imagenette "
                        "images taken from ImageNet's validation set, whose filenames carry no "
                        "wnid, so the filename-prefix fallback lumped them together. Put "
                        "noisy_imagenette.csv next to train/ and val/ (it ships in the "
                        "imagenette2 tarball) or set labels_csv in the config; the CSV labels "
                        "every image correctly.")
            raise RuntimeError(
                f"{root_dir} contains classes absent from the reference class list: "
                f"{sorted(unknown)}.{hint}"
            )

        self.samples = [(p, self.class_to_idx[k]) for p, k in samples]
        # Human-readable names, when we recognize the keys (imagenette wnids).
        self.class_names = class_names or [IMAGENETTE_CLASSES.get(c, c) for c in self.classes]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'idx': idx, 'label': label}


def build_image_transform(dataset_name, resize_img=256):
    """The exact preprocessing the autoencoders were trained with, factored out so the
    latent-flow pipeline cannot drift from it (a different normalization would put the
    frozen encoder off-distribution)."""
    transform_list = []
    if resize_img != -1:
        transform_list.append(transforms.Resize((resize_img, resize_img)))
    transform_list.append(transforms.ToTensor())
    if dataset_name.lower() == 'cifar10':
        transform_list.append(transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
    elif dataset_name.lower() == "imagenette":
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    if dataset_name.lower() == 'mnist':
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))
    return transforms.Compose(transform_list)


def get_labeled_datasets(dataset_name, path='./datasets', resize_img=256, val_ratio=0.2, seed=42,
                         labels_csv=None, label_column="noisy_labels_0"):
    """(trainset, valset) with real class labels, for class-conditional generation.

    imagenette -> LabeledImageDataset over <path>/train and <path>/val (any supported layout;
        noisy_imagenette.csv is auto-discovered next to them, and pass labels_csv to override).
    cifar10 / mnist -> the torchvision splits, which already carry labels.
    """
    transform = build_image_transform(dataset_name, resize_img)

    if dataset_name.lower() == "imagenette":
        train_path = os.path.join(path, "train")
        val_path = os.path.join(path, "val")
        trainset = LabeledImageDataset(train_path, transform=transform, labels_csv=labels_csv,
                                       label_column=label_column)
        # Force the val split onto the train split's class indices.
        valset = LabeledImageDataset(val_path, transform=transform, classes=trainset.classes,
                                     labels_csv=labels_csv, label_column=label_column)
        return trainset, valset

    if dataset_name.lower() not in ('cifar10', 'mnist'):
        raise ValueError(f"Unsupported dataset for conditional generation: {dataset_name}")

    dataset_class = torchvision.datasets.CIFAR10 if dataset_name.lower() == 'cifar10' else torchvision.datasets.MNIST
    full = dataset_class(root=path, train=True, download=True, transform=transform)
    total_len = len(full)
    val_len = int(total_len * val_ratio)
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full, [total_len - val_len, val_len], generator=generator)
    trainset, valset = TorchvisionDictWrapper(train_subset), TorchvisionDictWrapper(val_subset)
    trainset.classes = list(getattr(full, 'classes', range(10)))
    valset.classes = trainset.classes
    trainset.class_names = valset.class_names = [str(c) for c in trainset.classes]
    return trainset, valset


def get_benchmark_dataset(dataset_name, path='./datasets', split='train', val_ratio=0.2, resize_img=250, seed=42):
    """
    Downloads and prepares CIFAR10 or MNIST with Train/Val/Test splits.
    """
    # 1. Define standard transforms
    transform = build_image_transform(dataset_name, resize_img)

    # 2. Select the dataset class
    if dataset_name.lower() == "imagenette":
        # Paths pointing to your flattened directories
        train_path = os.path.join(path,"train")
        assert os.path.exists(train_path), f"Train path does not exist: {train_path}"
        val_path = os.path.join(path, "val")
        assert os.path.exists(val_path), f"Val path does not exist: {val_path}"
        train_dataset = FlatImageDataset(root_dir=train_path, transform=transform)
        val_dataset = FlatImageDataset(root_dir=val_path, transform=transform)

        return train_dataset, val_dataset
    elif dataset_name.lower() == 'cifar10':
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