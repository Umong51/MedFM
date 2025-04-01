import random
from typing import Callable, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as tf
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split

# Constants
FILE_PATH = "/home/mila/i/islamria/scratch/lidc_data/preproc/data_lidc.hdf5"
NUM_CLASSES = 2
RESOLUTION = 128
BACKGROUND_CLASS: Optional[int] = None


def one_hot_encoding(arr: np.ndarray) -> np.ndarray:
    """Converts a 2D array into a one-hot encoded 3D array."""
    res = np.zeros(arr.shape + (NUM_CLASSES,), dtype=np.float32)
    h, w = np.ix_(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    res[h, w, arr] = 1.0
    return res


def to_tensor(arr: np.ndarray) -> Tensor:
    """Converts a NumPy array to a PyTorch tensor, ensuring correct dimensions."""
    if arr.ndim == 2:
        arr = arr[:, :, None]  # Add a channel dimension if missing
    return torch.from_numpy(arr.transpose((2, 0, 1))).contiguous()


def transform(image: np.ndarray, labels: np.ndarray) -> Tuple[Tensor, Tensor]:
    """Apply data augmentation to both images and labels."""
    labels = labels.astype(int)
    labels = one_hot_encoding(labels)

    image_tensor = tf.to_tensor(image.transpose((1, 2, 0))).float()
    labels_tensor = to_tensor(labels)

    # Random horizontal and vertical flips
    if random.random() < 0.5:
        image_tensor = tf.hflip(image_tensor)
        labels_tensor = tf.hflip(labels_tensor)

    if random.random() < 0.5:
        image_tensor = tf.vflip(image_tensor)
        labels_tensor = tf.vflip(labels_tensor)

    # Random 90-degree rotations
    rots = np.random.randint(0, 4)
    image_tensor = torch.rot90(image_tensor, rots, [1, 2])
    labels_tensor = torch.rot90(labels_tensor, rots, [1, 2])

    image_tensor *= 2  # Normalize the image
    return image_tensor, labels_tensor


def test_transform(image, labels):
    """Prepare the image and labels for testing without random transformations."""
    image_tensor = tf.to_tensor(image.transpose((1, 2, 0))).float() * 2

    for i in range(4):
        labels[str(i)] = labels[str(i)].astype(int)
        labels[str(i)] = one_hot_encoding(labels[str(i)])
        labels[str(i)] = to_tensor(labels[str(i)])

    labels_tensor = torch.stack([labels[str(i)] for i in range(4)])
    return image_tensor, labels_tensor


class LIDC(Dataset):
    def __init__(self, dataset, transform: Optional[Callable] = None):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index: int):
        image = np.expand_dims(self.dataset["images"][index], axis=0)

        # Randomly select one rater's label
        rater_idx = random.randint(0, 3)
        label = self.dataset["labels"][index][rater_idx].astype(np.float32)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self) -> int:
        return len(self.dataset["images"])


class TestLIDC(Dataset):
    def __init__(self, dataset, transform: Callable):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index: int):
        image = np.expand_dims(self.dataset["images"][index], axis=0)

        # Select the four raters' labels for this image
        labels = {str(i): self.dataset["labels"][index][i] for i in range(4)}

        if self.transform:
            image, labels = self.transform(image, labels)

        return image, labels, np.array([0.25, 0.25, 0.25, 0.25])

    def __len__(self) -> int:
        return len(self.dataset["images"])


def training_dataset() -> LIDC:
    """Returns the training dataset with transformations."""
    dataset = LIDC(h5py.File(FILE_PATH, "r")["train"], transform)
    return dataset


def validation_dataset(max_size: Optional[int] = None) -> Union[TestLIDC, Subset]:
    """Returns the validation dataset, optionally limited in size."""
    dataset = TestLIDC(h5py.File(FILE_PATH, "r")["val"], test_transform)

    if max_size is None:
        return dataset

    generator = torch.Generator().manual_seed(1)
    dataset, _ = random_split(dataset, [max_size, len(dataset) - max_size], generator=generator)
    return dataset


def test_dataset(max_size: Optional[int] = None) -> Union[TestLIDC, Subset]:
    """Returns the test dataset, optionally limited in size."""
    dataset = TestLIDC(h5py.File(FILE_PATH, "r")["test"], test_transform)

    if max_size is None:
        return dataset

    return Subset(dataset, range(max_size))


def get_num_classes() -> int:
    """Returns the number of classes in the dataset."""
    return NUM_CLASSES


def get_ignore_class() -> Optional[int]:
    """Returns the background class (if any) to be ignored."""
    return BACKGROUND_CLASS
