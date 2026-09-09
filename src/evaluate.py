"""
Evaluation and visualization utilities for DCGAN.

This module provides functions for:
1. Generating sample grids from trained models
2. Creating interpolation animations
3. Computing inception score (optional)
4. Visualizing training progress
"""

import os
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from models import Generator


def load_generator(checkpoint_path, config, device):
    """
    Load generator from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Target device
    
    Returns:
        Loaded generator model
    """
    netG = Generator(
        nz=config['nz'],
        ngf=config['ngf'],
        nc=config['nc'],
        ngpu=config['ngpu']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netG.eval()
    
    return netG


def generate_grid(netG, nz, device, n_samples=64, nrow=8, save_path=None):
    """
    Generate a grid of samples.
    
    Args:
        netG: Generator model
        nz: Latent vector size
        device: Device
        n_samples: Number of samples to generate
        nrow: Number of images per row
        save_path: Optional path to save grid
    
    Returns:
        Grid tensor
    """
    with torch.no_grad():
        noise = torch.randn(n_samples, nz, 1, 1, device=device)
        fake = netG(noise).detach().cpu()
    
    grid = vutils.make_grid(fake, nrow=nrow, normalize=True, padding=2)
    
    if save_path:
        vutils.save_image(grid, save_path, normalize=True)
        print(f"Grid saved to {save_path}")
    
    return grid


def generate_interpolation(netG, nz, device, steps=10, n_pairs=5, save_path=None):
    """
    Generate linear interpolation between random latent vectors.
    
    Args:
        netG: Generator model
        nz: Latent vector size
        device: Device
        steps: Number of interpolation steps
        n_pairs: Number of interpolation pairs
        save_path: Optional path to save
    
    Returns:
        List of grids (one per pair)
    """
    grids = []
    
    with torch.no_grad():
        for _ in range(n_pairs):
            # Random start and end points
            z1 = torch.randn(1, nz, 1, 1, device=device)
            z2 = torch.randn(1, nz, 1, 1, device=device)
            
            # Interpolate
            alphas = torch.linspace(0, 1, steps, device=device)
            interpolated = []
            
            for alpha in alphas:
                z = (1 - alpha) * z1 + alpha * z2
                img = netG(z)
                interpolated.append(img)
            
            interpolated = torch.cat(interpolated, dim=0)
            grid = vutils.make_grid(interpolated.cpu(), nrow=steps, normalize=True, padding=2)
            grids.append(grid)
            
            if save_path:
                pair_path = save_path.replace('.png', f'_pair_{len(grids)}.png')
                vutils.save_image(grid, pair_path, normalize=True)
    
    return grids


def create_sample_sheet(checkpoint_dir, config, device, epochs=None, save_path=None):
    """
    Create a composite image showing samples from multiple epochs.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        config: Configuration dictionary
        device: Device
        epochs: List of epochs to include (default: all checkpoints)
        save_path: Path to save composite image
    """
    if epochs is None:
        # Find all checkpoints
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        epochs = []
        for cp in checkpoints:
            if 'epoch_' in cp:
                epoch_num = int(cp.split('epoch_')[1].split('.')[0])
                epochs.append(epoch_num)
        epochs = sorted(epochs)
    
    n_rows = len(epochs)
    n_cols = 8
    all_images = []
    
    for epoch in epochs:
        cp_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pth')
        if not os.path.exists(cp_path):
            continue
        
        netG = load_generator(cp_path, config, device)
        
        with torch.no_grad():
            noise = torch.randn(n_cols, config['nz'], 1, 1, device=device)
            fake = netG(noise).detach().cpu()
        
        all_images.append(fake)
    
    if not all_images:
        print("No valid checkpoints found")
        return
    
    # Create composite grid
    all_images = torch.cat(all_images, dim=0)
    grid = vutils.make_grid(all_images, nrow=n_cols, normalize=True, padding=2)
    
    if save_path:
        vutils.save_image(grid, save_path, normalize=True)
        print(f"Sample sheet saved to {save_path}")
    
    return grid


def plot_losses_from_log(log_dir, save_path=None):
    """
    Plot losses from TensorBoard logs (requires tensorboard backend).
    Alternative: parse CSV logs if available.
    """
    # This would require tensorboard data parsing
    # For simplicity, we'll note that losses are plotted during training
    pass


def save_individual_samples(netG, nz, device, n_samples=100, save_dir='samples/individual'):
    """
    Save individual generated samples as separate image files.
    
    Args:
        netG: Generator model
        nz: Latent vector size
        device: Device
        n_samples: Number of samples
        save_dir: Directory to save images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        noise = torch.randn(n_samples, nz, 1, 1, device=device)
        fake = netG(noise).detach().cpu()
    
    for i in range(n_samples):
        img = fake[i]
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = vutils.make_grid(img.unsqueeze(0), normalize=False, padding=0)
        save_path = os.path.join(save_dir, f'sample_{i:04d}.png')
        vutils.save_image(img, save_path)
    
    print(f"Saved {n_samples} individual samples to {save_dir}")


def analyze_mode_collapse(samples_dir, n_samples=1000):
    """
    Basic analysis for mode collapse detection.
    
    Computes pairwise distances between generated samples to detect
    if the generator is producing similar images (mode collapse).
    
    Args:
        samples_dir: Directory with generated samples
        n_samples: Number of samples to analyze
    
    Returns:
        Dictionary with statistics
    """
    # Load samples
    sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.png')])[:n_samples]
    
    if len(sample_files) < 2:
        return {'error': 'Not enough samples'}
    
    images = []
    for f in sample_files:
        img = Image.open(os.path.join(samples_dir, f)).convert('L')
        img = np.array(img).flatten() / 255.0
        images.append(img)
    
    images = np.array(images)
    
    # Compute pairwise distances (sample a subset for efficiency)
    n = min(100, len(images))
    indices = np.random.choice(len(images), n, replace=False)
    subset = images[indices]
    
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(subset[i] - subset[j])
            distances.append(dist)
    
    distances = np.array(distances)
    
    stats = {
        'mean_distance': float(distances.mean()),
        'std_distance': float(distances.std()),
        'min_distance': float(distances.min()),
        'max_distance': float(distances.max()),
        'num_samples_analyzed': n
    }
    
    print("Mode Collapse Analysis:")
    print(f"  Mean pairwise distance: {stats['mean_distance']:.4f}")
    print(f"  Std pairwise distance: {stats['std_distance']:.4f}")
    print(f"  Min distance: {stats['min_distance']:.4f}")
    print(f"  Max distance: {stats['max_distance']:.4f}")
    
    # Low mean distance with low std suggests mode collapse
    if stats['mean_distance'] < 0.1 and stats['std_distance'] < 0.05:
        print("  WARNING: Possible mode collapse detected!")
        stats['mode_collapse_warning'] = True
    else:
        stats['mode_collapse_warning'] = False
    
    return stats


if __name__ == '__main__':
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description='DCGAN Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.json', help='Config file')
    parser.add_argument('--n_samples', type=int, default=64, help='Number of samples')
    parser.add_argument('--output', type=str, default='samples/eval_grid.png', help='Output path')
    parser.add_argument('--interpolation', action='store_true', help='Generate interpolations')
    parser.add_argument('--sample_sheet', action='store_true', help='Create sample sheet from all checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.sample_sheet:
        create_sample_sheet(args.checkpoint_dir, config, device, save_path='samples/sample_sheet.png')
    elif args.interpolation:
        netG = load_generator(args.checkpoint, config, device)
        generate_interpolation(netG, config['nz'], device, save_path='samples/interpolation.png')
    else:
        netG = load_generator(args.checkpoint, config, device)
        generate_grid(netG, config['nz'], device, args.n_samples, save_path=args.output)