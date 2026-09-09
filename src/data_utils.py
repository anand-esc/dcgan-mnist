"""
Data utilities for DCGAN training.

Handles downloading and preprocessing of MNIST and Fashion-MNIST datasets.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_transform(image_size=64):
    """
    Get image transformations for DCGAN.
    
    DCGAN requires images to be normalized to [-1, 1] range because
    the generator uses Tanh activation in the output layer.
    
    Args:
        image_size: Target image size (default: 64)
    
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])


def get_dataset(dataset_name, data_root, transform, download=True):
    """
    Get dataset by name.
    
    Args:
        dataset_name: 'mnist' or 'fashion-mnist'
        data_root: Root directory for dataset
        transform: Transform to apply
        download: Whether to download if not present
    
    Returns:
        Dataset object
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=True,
            download=download,
            transform=transform
        )
    elif dataset_name == 'fashion-mnist' or dataset_name == 'fashion_mnist':
        dataset = torchvision.datasets.FashionMNIST(
            root=data_root,
            train=True,
            download=download,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'mnist' or 'fashion-mnist'")
    
    return dataset


def get_dataloader(config):
    """
    Create DataLoader for training.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        DataLoader object
    """
    transform = get_transform(config['image_size'])
    
    dataset = get_dataset(
        config['dataset'],
        config['data_root'],
        transform,
        download=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    return dataloader


def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1] for visualization.
    
    Args:
        tensor: Input tensor in [-1, 1] range
    
    Returns:
        Tensor in [0, 1] range
    """
    return (tensor + 1) / 2


def get_class_names(dataset_name):
    """Get class names for dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        return [str(i) for i in range(10)]
    elif dataset_name == 'fashion-mnist' or dataset_name == 'fashion_mnist':
        return [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    return [str(i) for i in range(10)]


if __name__ == '__main__':
    # Test data loading
    import json
    
    with open('configs/config.json', 'r') as f:
        config = json.load(f)
    
    dataloader = get_dataloader(config)
    
    print(f"Dataset: {config['dataset']}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {config['batch_size']}")
    
    # Check a batch
    for images, labels in dataloader:
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Labels: {labels[:10]}")
        break