"""Standalone training script for pure image-to-image translation using Swin-V2 UNet.

This script trains ONLY the generator using L1 loss (no GAN, no discriminator).
It uses the SwinV2UNet256Generator for 256x256 image-to-image translation.

Example usage:
CUDA_VISIBLE_DEVICES=5 python train_pure_swinv2.py \
    --dataroot /path/to/data/AB \
    --checkpoints_dir /path/to/checkpoints \
    --name pure_swinv2 \
    --input_nc 1 --output_nc 1 \
    --n_epochs 100 --n_epochs_decay 0 \
    --lr 0.0002 --save_epoch_freq 1
"""

import argparse
import time
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import lr_scheduler

# Import from existing codebase
from data import create_dataset
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp, mkdirs
from models.swin_v2_unet_generator import SwinV2UNet256Generator

# Enable cuDNN auto-tuning for faster convolution operations
torch.backends.cudnn.benchmark = True

# ============================================================================
# Argument Parser
# ============================================================================

def get_options():
    """Parse command line arguments for pure img2img training."""
    parser = argparse.ArgumentParser(
        description='Pure Image-to-Image Training with SwinV2 UNet (L1 loss only)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configurable via CLI
    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders train)')
    parser.add_argument('--checkpoints_dir', required=True, help='models are saved here')
    parser.add_argument('--name', required=True, help='name of the experiment')
    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
    parser.add_argument('--n_epochs', type=int, default=100, help='# of epochs with initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=0, help='# of epochs to linearly decay lr to zero')
    parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at end of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

    opt = parser.parse_args()

    # Hardcoded values
    opt.batch_size = 1
    opt.direction = 'AtoB'
    opt.dataset_mode = 'aligned'
    opt.preprocess = 'none'
    opt.no_flip = True
    opt.load_size = 256
    opt.crop_size = 256
    opt.beta1 = 0.5
    opt.beta2 = 0.999
    opt.lr_policy = 'linear'
    opt.print_freq = 100
    opt.display_freq = 400
    opt.save_latest_freq = 5000
    opt.no_html = True
    opt.num_threads = 4
    opt.serial_batches = False
    opt.max_dataset_size = float("inf")
    opt.phase = 'train'
    opt.epoch_count = 1
    opt.isTrain = True
    opt.display_winsize = 256
    opt.update_html_freq = 1000
    opt.use_wandb = False
    opt.wandb_project_name = 'SwinV2-Img2Img'
    opt.verbose = False

    return opt


# ============================================================================
# Learning Rate Scheduler
# ============================================================================

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler.

    For 'linear', keep same lr for first n_epochs, then linearly decay to zero.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        raise NotImplementedError(f'learning rate policy [{opt.lr_policy}] is not implemented')
    return scheduler


# ============================================================================
# Model Class
# ============================================================================

class PureImg2ImgModel:
    """Simple image-to-image translation model with L1 loss only.

    Uses SwinV2UNet256Generator for generation and L1 loss for supervision.
    No discriminator, no GAN loss.
    """

    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.save_dir = Path(opt.checkpoints_dir) / opt.name
        mkdirs(str(self.save_dir))

        # Define generator (auto-downloads pretrained weights if needed)
        self.netG = SwinV2UNet256Generator(
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            depths_decoder=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            use_dropout=False,
            drop_rate=0.0,
            pretrained_path="checkpoints/swinv2_tiny_patch4_window8_256.pth"
        )
        self.netG.to(device)

        # Wrap with DDP if distributed
        if dist.is_initialized():
            self.netG = torch.nn.parallel.DistributedDataParallel(
                self.netG, device_ids=[device.index] if device.index is not None else None
            )
            dist.barrier()

        # Define loss function
        self.criterionL1 = nn.L1Loss()

        # Define optimizer (same as pix2pix: Adam with beta1=0.5, beta2=0.999)
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(),
            lr=opt.lr,
            betas=(opt.beta1, opt.beta2)
        )

        # Define scheduler
        self.scheduler = get_scheduler(self.optimizer_G, opt)

        # Names for logging
        self.loss_names = ['L1']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G']

        # Storage for current batch
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.loss_L1 = None
        self.image_paths = None

        self._print_network_info()

    def _print_network_info(self):
        """Print network parameter count."""
        net = self.netG.module if hasattr(self.netG, 'module') else self.netG
        num_params = sum(p.numel() for p in net.parameters())
        print(f'[Network G] Total parameters: {num_params / 1e6:.3f} M')

    def set_input(self, data):
        """Unpack input data from the dataloader."""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = data['A' if AtoB else 'B'].to(self.device)
        self.real_B = data['B' if AtoB else 'A'].to(self.device)
        self.image_paths = data['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass."""
        self.fake_B = self.netG(self.real_A)

    def backward(self):
        """Calculate L1 loss and backpropagate."""
        self.loss_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_L1.backward()

    def optimize_parameters(self):
        """Forward pass, compute loss, and update weights."""
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization."""
        pass  # No additional computation needed

    def update_learning_rate(self):
        """Update learning rate at end of epoch."""
        old_lr = self.optimizer_G.param_groups[0]['lr']
        self.scheduler.step()
        new_lr = self.optimizer_G.param_groups[0]['lr']
        print(f'Learning rate: {old_lr:.7f} -> {new_lr:.7f}')

    def get_current_losses(self):
        """Return current losses as OrderedDict."""
        return OrderedDict([('L1', float(self.loss_L1))])

    def get_current_visuals(self):
        """Return current visuals as OrderedDict."""
        return OrderedDict([
            ('real_A', self.real_A),
            ('fake_B', self.fake_B),
            ('real_B', self.real_B)
        ])

    def get_image_paths(self):
        """Return image paths."""
        return self.image_paths

    def save_networks(self, epoch):
        """Save generator network to disk."""
        # Only save on rank 0 for distributed
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        save_filename = f'{epoch}_net_G.pth'
        save_path = self.save_dir / save_filename

        net = self.netG.module if hasattr(self.netG, 'module') else self.netG
        # Handle torch.compile if used
        if hasattr(net, '_orig_mod'):
            net = net._orig_mod

        torch.save(net.state_dict(), save_path)
        print(f'Saved model to {save_path}')

    def load_networks(self, epoch):
        """Load generator network from disk."""
        load_filename = f'{epoch}_net_G.pth'
        load_path = self.save_dir / load_filename

        net = self.netG.module if hasattr(self.netG, 'module') else self.netG
        print(f'Loading model from {load_path}')

        state_dict = torch.load(load_path, map_location=str(self.device))
        net.load_state_dict(state_dict)

        if dist.is_initialized():
            dist.barrier()

    def eval(self):
        """Set model to evaluation mode."""
        self.netG.eval()

    def train(self):
        """Set model to training mode."""
        self.netG.train()


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    # Parse options
    opt = get_options()

    # Initialize device (single GPU or DDP)
    opt.device = init_ddp()

    # Print options
    print('------------ Options -------------')
    for k, v in sorted(vars(opt).items()):
        print(f'{k}: {v}')
    print('-------------- End ----------------')

    # Save options to file
    expr_dir = Path(opt.checkpoints_dir) / opt.name
    mkdirs(str(expr_dir))
    opt_file = expr_dir / f'{opt.phase}_opt.txt'
    with open(opt_file, 'w') as f:
        for k, v in sorted(vars(opt).items()):
            f.write(f'{k}: {v}\n')

    # Create dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'The number of training images = {dataset_size}')

    # Create model
    model = PureImg2ImgModel(opt, opt.device)

    # Create visualizer
    visualizer = Visualizer(opt)

    # Training loop
    total_iters = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        # Set epoch for DistributedSampler
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # Forward, loss, backward
            model.set_input(data)
            model.optimize_parameters()

            # Display results
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            # Print losses
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            # Save latest model
            if total_iters % opt.save_latest_freq == 0:
                print(f'Saving latest model (epoch {epoch}, total_iters {total_iters})')
                model.save_networks('latest')

            iter_data_time = time.time()

        # Update learning rate
        model.update_learning_rate()

        # Save model at end of epoch
        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving model at end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)

        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec')

    # Cleanup
    cleanup_ddp()
    print('Training completed.')


if __name__ == '__main__':
    main()
