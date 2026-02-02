"""
Swin Transformer model for supervised MR-to-CT synthesis.

Training uses L1 loss only (no GAN discriminator).
Uses AdamW optimizer with warmup + cosine decay schedule.
"""

import torch
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict

from .swin_networks import SwinGenerator


class SwinModel:
    """Swin Transformer model for MR-to-CT synthesis.

    Attributes:
        loss_names: List of loss names for logging
        model_names: List of model names for saving
        visual_names: List of image names for visualization
    """

    def __init__(self, opt):
        """Initialize the model.

        Args:
            opt: Options object with training configuration
        """
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = opt.device

        # Setup save directory
        self.save_dir = Path(opt.checkpoints_dir) / opt.name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Define tracking lists
        self.loss_names = ['L1']
        self.model_names = ['G']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.optimizers = []
        self.schedulers = []

        # Create generator
        pretrained = getattr(opt, 'pretrained', True)
        self.netG = SwinGenerator(
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            pretrained=pretrained
        ).to(self.device)

        # Print parameter count
        n_params = sum(p.numel() for p in self.netG.parameters())
        print(f'[SwinModel] Generator parameters: {n_params:,}')

        if self.isTrain:
            # Loss function
            self.criterionL1 = nn.L1Loss()

            # Optimizer - AdamW is standard for transformers
            self.optimizer_G = torch.optim.AdamW(
                self.netG.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, opt.beta2),
                weight_decay=opt.weight_decay
            )
            self.optimizers.append(self.optimizer_G)

            # Learning rate scheduler with warmup
            self.scheduler = self._create_scheduler(opt)
            self.schedulers.append(self.scheduler)

            # Gradient clipping value
            self.grad_clip = getattr(opt, 'grad_clip', 1.0)

            # Loss weight
            self.lambda_L1 = getattr(opt, 'lambda_L1', 100.0)

    def _create_scheduler(self, opt):
        """Create warmup + cosine annealing scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_epochs = getattr(opt, 'warmup_epochs', 10)
        n_epochs = opt.n_epochs

        warmup_scheduler = LinearLR(
            self.optimizer_G,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer_G,
            T_max=n_epochs - warmup_epochs,
            eta_min=opt.lr * 0.01
        )
        scheduler = SequentialLR(
            self.optimizer_G,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        return scheduler

    def set_input(self, input):
        """Unpack input data from dataloader.

        Args:
            input: Dictionary with 'A', 'B', 'A_paths', 'B_paths'
        """
        AtoB = getattr(self.opt, 'direction', 'AtoB') == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass."""
        self.fake_B = self.netG(self.real_A)

    def backward_G(self):
        """Calculate L1 loss and backpropagate."""
        self.loss_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
        self.loss_L1.backward()

    def optimize_parameters(self):
        """Forward, backward, and optimize."""
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.netG.parameters(),
                self.grad_clip
            )

        self.optimizer_G.step()

    def update_learning_rate(self):
        """Update learning rate at end of epoch."""
        old_lr = self.optimizer_G.param_groups[0]['lr']
        self.scheduler.step()
        new_lr = self.optimizer_G.param_groups[0]['lr']
        print(f'Learning rate: {old_lr:.2e} -> {new_lr:.2e}')

    def get_current_losses(self):
        """Return current losses as OrderedDict."""
        return OrderedDict([('L1', self.loss_L1.item())])

    def get_current_visuals(self):
        """Return visualization images as OrderedDict."""
        return OrderedDict([
            ('real_A', self.real_A.detach()),
            ('fake_B', self.fake_B.detach()),
            ('real_B', self.real_B.detach())
        ])

    def save_networks(self, epoch):
        """Save model checkpoint.

        Args:
            epoch: Current epoch number or 'latest'
        """
        save_filename = f'{epoch}_net_G.pth'
        save_path = self.save_dir / save_filename
        torch.save(self.netG.state_dict(), save_path)

        # Also save optimizer state
        opt_filename = f'{epoch}_optimizer_G.pth'
        opt_path = self.save_dir / opt_filename
        torch.save({
            'optimizer': self.optimizer_G.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, opt_path)

        print(f'Saved checkpoint: {save_path}')

    def load_networks(self, epoch):
        """Load model checkpoint.

        Args:
            epoch: Epoch number or 'latest' to load
        """
        load_filename = f'{epoch}_net_G.pth'
        load_path = self.save_dir / load_filename

        state_dict = torch.load(load_path, map_location=self.device)
        self.netG.load_state_dict(state_dict)
        print(f'Loaded checkpoint: {load_path}')

        # Try to load optimizer state
        if self.isTrain:
            opt_filename = f'{epoch}_optimizer_G.pth'
            opt_path = self.save_dir / opt_filename
            if opt_path.exists():
                opt_state = torch.load(opt_path, map_location=self.device)
                self.optimizer_G.load_state_dict(opt_state['optimizer'])
                self.scheduler.load_state_dict(opt_state['scheduler'])
                print(f'Loaded optimizer state: {opt_path}')

    def eval(self):
        """Set model to evaluation mode."""
        self.netG.eval()

    def train(self):
        """Set model to training mode."""
        self.netG.train()

    def test(self):
        """Run inference without gradients."""
        with torch.no_grad():
            self.forward()
