"""
DCGAN Project Package

Deep Convolutional Generative Adversarial Network implementation
for MNIST and Fashion-MNIST image generation.

Modules:
- models: Generator and Discriminator architectures
- train: Training loop with stabilization techniques
- evaluate: Evaluation and visualization utilities
- data_utils: Data loading and preprocessing
"""

__version__ = '1.0.0'
__author__ = 'DCGAN Project'

from .models import Generator, Discriminator, weights_init
from .data_utils import get_dataloader, get_transform, get_dataset

__all__ = [
    'Generator',
    'Discriminator', 
    'weights_init',
    'get_dataloader',
    'get_transform',
    'get_dataset'
]