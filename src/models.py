"""
DCGAN Model Architecture

This module implements the Generator and Discriminator networks for DCGAN
following the architecture guidelines from the original DCGAN paper:
- Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation 
  Learning with Deep Convolutional Generative Adversarial Networks.

Key architectural decisions:
1. Replace pooling layers with strided convolutions (discriminator) and 
   fractional-strided convolutions (generator)
2. Use BatchNorm in both generator and discriminator
3. Remove fully connected hidden layers for deeper architectures
4. Use ReLU activation in generator (except output which uses Tanh)
5. Use LeakyReLU activation in discriminator
"""

import torch
import torch.nn as nn


def weights_init(m):
    """
    Custom weight initialization for DCGAN.
    
    From the DCGAN paper: "All weights were initialized from a zero-centered 
    Normal distribution with standard deviation 0.02."
    
    Args:
        m: Module to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Generator Network for DCGAN.
    
    Architecture: 
    Input: Z (latent vector) of size nz x 1 x 1
    -> ConvTranspose2d(nz, ngf*8, 4, 1, 0) -> BatchNorm -> ReLU
    -> ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1) -> BatchNorm -> ReLU
    -> ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1) -> BatchNorm -> ReLU
    -> ConvTranspose2d(ngf*2, ngf, 4, 2, 1) -> BatchNorm -> ReLU
    -> ConvTranspose2d(ngf, nc, 4, 2, 1) -> Tanh
    Output: Image of size nc x 64 x 64
    
    Args:
        nz (int): Size of latent vector (default: 100)
        ngf (int): Number of generator features (default: 64)
        nc (int): Number of channels in output image (default: 1 for MNIST)
        ngpu (int): Number of GPUs available (default: 1)
    """
    
    def __init__(self, nz=100, ngf=64, nc=1, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        
        self.main = nn.Sequential(
            # Input: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: nc x 64 x 64
        )
    
    def forward(self, input):
        """
        Forward pass of the generator.
        
        Args:
            input (torch.Tensor): Latent vector of shape (batch_size, nz, 1, 1)
            
        Returns:
            torch.Tensor: Generated images of shape (batch_size, nc, 64, 64)
        """
        if input.dim() == 2:
            input = input.unsqueeze(2).unsqueeze(3)
        return self.main(input)


class Discriminator(nn.Module):
    """
    Discriminator Network for DCGAN.
    
    Architecture:
    Input: Image of size nc x 64 x 64
    -> Conv2d(nc, ndf, 4, 2, 1) -> LeakyReLU(0.2)
    -> Conv2d(ndf, ndf*2, 4, 2, 1) -> BatchNorm -> LeakyReLU(0.2)
    -> Conv2d(ndf*2, ndf*4, 4, 2, 1) -> BatchNorm -> LeakyReLU(0.2)
    -> Conv2d(ndf*4, ndf*8, 4, 2, 1) -> BatchNorm -> LeakyReLU(0.2)
    -> Conv2d(ndf*8, 1, 4, 1, 0) -> Sigmoid
    Output: Single scalar (probability of real)
    
    Args:
        nc (int): Number of channels in input image (default: 1 for MNIST)
        ndf (int): Number of discriminator features (default: 64)
        ngpu (int): Number of GPUs available (default: 1)
    """
    
    def __init__(self, nc=1, ndf=64, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        self.main = nn.Sequential(
            # Input: nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: ndf x 32 x 32
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 16 x 16
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 8 x 8
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 4 x 4
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, input):
        """
        Forward pass of the discriminator.
        
        Args:
            input (torch.Tensor): Input images of shape (batch_size, nc, 64, 64)
            
        Returns:
            torch.Tensor: Probability scores of shape (batch_size, 1, 1, 1)
        """
        return self.main(input).view(-1, 1).squeeze(1)


def get_model_summary(model, input_size):
    """
    Print model architecture summary.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input tensor size (C, H, W) or (N, C, H, W)
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = {
                "input_shape": list(input[0].size()) if input else [],
                "output_shape": list(output.size()) if not isinstance(output, tuple) else [list(o.size()) for o in output],
                "num_params": sum(p.numel() for p in module.parameters())
            }
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and module != model:
            hooks.append(module.register_forward_hook(hook))
    
    summary = {}
    hooks = []
    model.apply(register_hook)
    
    if len(input_size) == 3:
        x = torch.randn(1, *input_size)
    else:
        x = torch.randn(*input_size)
    
    model(x)
    
    for h in hooks:
        h.remove()
    
    print("-" * 80)
    print(f"{'Layer':<30} {'Input Shape':<20} {'Output Shape':<20} {'Params':>10}")
    print("-" * 80)
    total_params = 0
    for layer, info in summary.items():
        in_shape = str(info['input_shape'])
        out_shape = str(info['output_shape'])
        params = info['num_params']
        total_params += params
        print(f"{layer:<30} {in_shape:<20} {out_shape:<20} {params:>10,}")
    print("-" * 80)
    print(f"Total Parameters: {total_params:,}")
    print("-" * 80)


if __name__ == "__main__":
    # Test model architectures
    print("=" * 80)
    print("GENERATOR ARCHITECTURE")
    print("=" * 80)
    netG = Generator(nz=100, ngf=64, nc=1)
    netG.apply(weights_init)
    get_model_summary(netG, (100, 1, 1))
    
    print("\n" + "=" * 80)
    print("DISCRIMINATOR ARCHITECTURE")
    print("=" * 80)
    netD = Discriminator(nc=1, ndf=64)
    netD.apply(weights_init)
    get_model_summary(netD, (1, 64, 64))
    
    # Test forward pass
    print("\nTesting forward passes...")
    z = torch.randn(16, 100, 1, 1)
    fake_img = netG(z)
    print(f"Generator output shape: {fake_img.shape}")
    
    real_img = torch.randn(16, 1, 64, 64)
    output = netD(real_img)
    print(f"Discriminator output shape: {output.shape}")
    
    output_fake = netD(fake_img)
    print(f"Discriminator output (fake) shape: {output_fake.shape}")