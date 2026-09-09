"""
DCGAN Training Script

This module implements the training loop for DCGAN with various stabilization techniques:
1. Label Smoothing: Using soft labels (0.9 for real, 0.1 for fake) instead of hard labels (1.0, 0.0)
2. Batch Normalization: Applied in both generator and discriminator (already in model)
3. Tuned Learning Rates: Using Adam optimizer with beta1=0.5, beta2=0.999
4. Minimax Objective: Standard GAN loss with BCE

The minimax training objective:
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]

Where:
- D(x) is the discriminator's estimate of probability that real data x is real
- D(G(z)) is the discriminator's estimate of probability that generated data G(z) is real
- Generator tries to minimize log(1 - D(G(z))) which is equivalent to maximizing log D(G(z))
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import Generator, Discriminator, weights_init
from data_utils import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='DCGAN Training')
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def setup_device(config, device_arg):
    if device_arg:
        device = torch.device(device_arg)
    elif config.get('ngpu', 1) > 0 and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_models(config, device):
    """Initialize Generator and Discriminator with weight initialization."""
    netG = Generator(
        nz=config['nz'],
        ngf=config['ngf'],
        nc=config['nc'],
        ngpu=config['ngpu']
    ).to(device)
    
    netD = Discriminator(
        nc=config['nc'],
        ndf=config['ndf'],
        ngpu=config['ngpu']
    ).to(device)
    
    # Handle multi-GPU
    if config['ngpu'] > 1 and torch.cuda.device_count() > 1:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
    
    # Apply weight initialization
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    return netG, netD


def create_optimizers(netG, netD, config):
    """Create Adam optimizers with tuned hyperparameters."""
    optimizerG = optim.Adam(
        netG.parameters(),
        lr=config['lr'],
        betas=(config['beta1'], config['beta2'])
    )
    optimizerD = optim.Adam(
        netD.parameters(),
        lr=config['lr'],
        betas=(config['beta1'], config['beta2'])
    )
    return optimizerG, optimizerD


def create_loss_function():
    """Binary Cross Entropy loss for GAN training."""
    return nn.BCELoss()


def get_labels(batch_size, device, label_smoothing=0.0, real=True):
    """
    Create labels with optional label smoothing.
    
    Label smoothing prevents the discriminator from becoming too confident,
    which helps prevent gradient vanishing and mode collapse.
    
    Args:
        batch_size: Number of samples
        device: Target device
        label_smoothing: Smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        real: True for real labels, False for fake labels
    
    Returns:
        Tensor of labels
    """
    if real:
        # Real labels: 1.0 - smoothing (e.g., 0.9 instead of 1.0)
        labels = torch.full((batch_size,), 1.0 - label_smoothing, device=device)
    else:
        # Fake labels: smoothing (e.g., 0.1 instead of 0.0)
        labels = torch.full((batch_size,), label_smoothing, device=device)
    return labels


def train_discriminator(netD, netG, optimizerD, criterion, real_data, config, device):
    """
    Train the discriminator for one step.
    
    Discriminator loss: L_D = -[log D(x) + log(1 - D(G(z)))]
    
    We maximize this by minimizing the negative, so we use BCE with:
    - Real labels: 1 (smoothed to 0.9)
    - Fake labels: 0 (smoothed to 0.1)
    """
    netD.zero_grad()
    batch_size = real_data.size(0)
    
    # Train with real images
    label_real = get_labels(batch_size, device, config['label_smoothing'], real=True)
    output_real = netD(real_data)
    errD_real = criterion(output_real, label_real)
    errD_real.backward()
    D_x = output_real.mean().item()
    
    # Train with fake images
    noise = torch.randn(batch_size, config['nz'], 1, 1, device=device)
    fake = netG(noise)
    label_fake = get_labels(batch_size, device, config['label_smoothing'], real=False)
    output_fake = netD(fake.detach())
    errD_fake = criterion(output_fake, label_fake)
    errD_fake.backward()
    D_G_z1 = output_fake.mean().item()
    
    errD = errD_real + errD_fake
    optimizerD.step()
    
    return errD.item(), D_x, D_G_z1


def train_generator(netD, netG, optimizerG, criterion, batch_size, config, device):
    """
    Train the generator for one step.
    
    Generator loss: L_G = -log D(G(z))
    
    We want the generator to fool the discriminator, so we use real labels (1.0)
    for the generated images. This is equivalent to maximizing log D(G(z)).
    """
    netG.zero_grad()
    
    # Generate fake images
    noise = torch.randn(batch_size, config['nz'], 1, 1, device=device)
    fake = netG(noise)
    
    # Generator wants discriminator to classify fake as real
    label_real = get_labels(batch_size, device, config['label_smoothing'], real=True)
    output = netD(fake)
    errG = criterion(output, label_real)
    errG.backward()
    D_G_z2 = output.mean().item()
    
    optimizerG.step()
    
    return errG.item(), D_G_z2


def save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, config, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'config': config
    }, path)


def load_checkpoint(path, netG, netD, optimizerG, optimizerD, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    epoch = checkpoint['epoch']
    config = checkpoint.get('config', {})
    return epoch, config


def generate_samples(netG, fixed_noise, device, epoch, sample_dir, nrow=8):
    """Generate and save sample images."""
    netG.eval()
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    netG.train()
    
    # Save grid
    grid_path = os.path.join(sample_dir, f'epoch_{epoch:04d}.png')
    vutils.save_image(fake, grid_path, normalize=True, nrow=nrow)
    return grid_path


def save_loss_plot(losses, save_path):
    """Save loss curves as image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(losses['G']) + 1)
    
    # Generator loss
    axes[0, 0].plot(epochs, losses['G'], 'b-', label='Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Discriminator loss
    axes[0, 1].plot(epochs, losses['D'], 'r-', label='Discriminator Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # D(x) and D(G(z))
    axes[1, 0].plot(epochs, losses['D_x'], 'g-', label='D(x) - Real')
    axes[1, 0].plot(epochs, losses['D_G_z1'], 'orange', label='D(G(z)) - Before G update')
    axes[1, 0].plot(epochs, losses['D_G_z2'], 'purple', label='D(G(z)) - After G update')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_title('Discriminator Outputs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined loss
    axes[1, 1].plot(epochs, losses['G'], 'b-', label='Generator')
    axes[1, 1].plot(epochs, losses['D'], 'r-', label='Discriminator')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Combined Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train(config, device, resume_path=None):
    """Main training loop."""
    
    # Setup directories
    os.makedirs(config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
    os.makedirs(config.get('sample_dir', 'samples'), exist_ok=True)
    os.makedirs(config.get('log_dir', 'logs'), exist_ok=True)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=config.get('log_dir', 'logs'))
    
    # Data loader
    dataloader = get_dataloader(config)
    
    # Models
    netG, netD = create_models(config, device)
    
    # Optimizers
    optimizerG, optimizerD = create_optimizers(netG, netD, config)
    
    # Loss function
    criterion = create_loss_function()
    
    # Fixed noise for consistent sample generation
    fixed_noise = torch.randn(64, config['nz'], 1, 1, device=device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_path:
        start_epoch, _ = load_checkpoint(resume_path, netG, netD, optimizerG, optimizerD, device)
        print(f"Resumed from epoch {start_epoch}")
    
    # Loss tracking
    losses = {
        'G': [],
        'D': [],
        'D_x': [],
        'D_G_z1': [],
        'D_G_z2': []
    }
    
    print("Starting training...")
    print(f"Dataset: {config['dataset']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Label smoothing: {config['label_smoothing']}")
    print(f"Beta1: {config['beta1']}, Beta2: {config['beta2']}")
    print("-" * 60)
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_G_loss = 0.0
        epoch_D_loss = 0.0
        epoch_D_x = 0.0
        epoch_D_G_z1 = 0.0
        epoch_D_G_z2 = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for i, (real_data, _) in enumerate(pbar):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Train Discriminator
            errD, D_x, D_G_z1 = train_discriminator(
                netD, netG, optimizerD, criterion, real_data, config, device
            )
            
            # Train Generator
            errG, D_G_z2 = train_generator(
                netD, netG, optimizerG, criterion, batch_size, config, device
            )
            
            # Accumulate losses
            epoch_G_loss += errG
            epoch_D_loss += errD
            epoch_D_x += D_x
            epoch_D_G_z1 += D_G_z1
            epoch_D_G_z2 += D_G_z2
            num_batches += 1
            
            # Log to tensorboard
            global_step = epoch * len(dataloader) + i
            if global_step % config['log_interval'] == 0:
                writer.add_scalar('Loss/Generator', errG, global_step)
                writer.add_scalar('Loss/Discriminator', errD, global_step)
                writer.add_scalar('Discriminator/D_x', D_x, global_step)
                writer.add_scalar('Discriminator/D_G_z1', D_G_z1, global_step)
                writer.add_scalar('Discriminator/D_G_z2', D_G_z2, global_step)
            
            # Generate samples
            if global_step % config['sample_interval'] == 0:
                generate_samples(netG, fixed_noise, device, global_step, config.get('sample_dir', 'samples'))
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{errG:.4f}',
                'D_loss': f'{errD:.4f}',
                'D(x)': f'{D_x:.4f}',
                'D(G(z))': f'{D_G_z2:.4f}'
            })
        
        # Average losses for epoch
        avg_G_loss = epoch_G_loss / num_batches
        avg_D_loss = epoch_D_loss / num_batches
        avg_D_x = epoch_D_x / num_batches
        avg_D_G_z1 = epoch_D_G_z1 / num_batches
        avg_D_G_z2 = epoch_D_G_z2 / num_batches
        
        losses['G'].append(avg_G_loss)
        losses['D'].append(avg_D_loss)
        losses['D_x'].append(avg_D_x)
        losses['D_G_z1'].append(avg_D_G_z1)
        losses['D_G_z2'].append(avg_D_G_z2)
        
        # Log epoch metrics
        writer.add_scalar('Epoch/Generator_Loss', avg_G_loss, epoch)
        writer.add_scalar('Epoch/Discriminator_Loss', avg_D_loss, epoch)
        writer.add_scalar('Epoch/D_x', avg_D_x, epoch)
        writer.add_scalar('Epoch/D_G_z1', avg_D_G_z1, epoch)
        writer.add_scalar('Epoch/D_G_z2', avg_D_G_z2, epoch)
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] "
              f"G_loss: {avg_G_loss:.4f} | D_loss: {avg_D_loss:.4f} | "
              f"D(x): {avg_D_x:.4f} | D(G(z)): {avg_D_G_z2:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(
                config.get('checkpoint_dir', 'checkpoints'),
                f'checkpoint_epoch_{epoch+1:04d}.pth'
            )
            save_checkpoint(netG, netD, optimizerG, optimizerD, epoch+1, config, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Generate epoch samples
        generate_samples(netG, fixed_noise, device, epoch+1, config.get('sample_dir', 'samples'))
    
    # Save final checkpoint
    final_path = os.path.join(config.get('checkpoint_dir', 'checkpoints'), 'checkpoint_final.pth')
    save_checkpoint(netG, netD, optimizerG, optimizerD, config['num_epochs'], config, final_path)
    print(f"Final checkpoint saved: {final_path}")
    
    # Save loss plot
    plot_path = os.path.join(config.get('sample_dir', 'samples'), 'loss_curves.png')
    save_loss_plot(losses, plot_path)
    print(f"Loss curves saved: {plot_path}")
    
    writer.close()
    return netG, netD, losses


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with defaults if not present
    config.setdefault('checkpoint_dir', 'checkpoints')
    config.setdefault('sample_dir', 'samples')
    config.setdefault('log_dir', 'logs')
    
    setup_seed(config.get('seed', 42))
    device = setup_device(config, args.device)
    
    train(config, device, args.resume)


if __name__ == '__main__':
    main()