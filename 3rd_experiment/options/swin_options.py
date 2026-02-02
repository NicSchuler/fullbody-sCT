"""
Options for Swin Transformer MR-to-CT synthesis.
"""

import argparse
import torch
from pathlib import Path


class SwinOptions:
    """Options class for Swin Transformer training."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Swin Transformer for MR-to-CT synthesis'
        )
        self.initialized = False

    def initialize(self):
        """Define all options."""
        # Data options
        self.parser.add_argument('--dataroot', type=str, required=True,
                                 help='Path to dataset (cyclegan format)')
        self.parser.add_argument('--phase', type=str, default='train',
                                 choices=['train', 'val', 'test'],
                                 help='train, val, or test')
        self.parser.add_argument('--max_dataset_size', type=int, default=float('inf'),
                                 help='Maximum number of samples')
        self.parser.add_argument('--direction', type=str, default='AtoB',
                                 help='AtoB or BtoA')

        # Model options
        self.parser.add_argument('--input_nc', type=int, default=1,
                                 help='Number of input channels')
        self.parser.add_argument('--output_nc', type=int, default=1,
                                 help='Number of output channels')
        self.parser.add_argument('--pretrained', action='store_true', default=True,
                                 help='Use pretrained SwinV2-T weights')
        self.parser.add_argument('--no_pretrained', action='store_false', dest='pretrained',
                                 help='Train from scratch')

        # Training options
        self.parser.add_argument('--n_epochs', type=int, default=200,
                                 help='Total number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=8,
                                 help='Batch size')
        self.parser.add_argument('--lr', type=float, default=1e-4,
                                 help='Initial learning rate')
        self.parser.add_argument('--beta1', type=float, default=0.9,
                                 help='Adam beta1')
        self.parser.add_argument('--beta2', type=float, default=0.999,
                                 help='Adam beta2')
        self.parser.add_argument('--weight_decay', type=float, default=0.05,
                                 help='Weight decay for AdamW')
        self.parser.add_argument('--warmup_epochs', type=int, default=10,
                                 help='Number of warmup epochs')
        self.parser.add_argument('--grad_clip', type=float, default=1.0,
                                 help='Gradient clipping value (0 to disable)')
        self.parser.add_argument('--lambda_L1', type=float, default=100.0,
                                 help='Weight for L1 loss')

        # Data augmentation
        self.parser.add_argument('--load_size', type=int, default=256,
                                 help='Scale images to this size')
        self.parser.add_argument('--crop_size', type=int, default=256,
                                 help='Crop to this size')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='Disable random horizontal flip')

        # Checkpoints
        self.parser.add_argument('--name', type=str, default='swin_experiment',
                                 help='Name of the experiment')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                                 help='Directory to save checkpoints')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10,
                                 help='Save checkpoint every N epochs')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='Continue training from checkpoint')
        self.parser.add_argument('--epoch', type=str, default='latest',
                                 help='Epoch to load for continue_train or test')

        # Logging
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='Print losses every N iterations')

        # Hardware
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='GPU IDs (comma-separated, e.g., 0,1)')
        self.parser.add_argument('--num_threads', type=int, default=4,
                                 help='Number of data loading threads')

        self.initialized = True

    def parse(self, args=None):
        """Parse and process options.

        Args:
            args: Optional list of arguments (for testing)

        Returns:
            Namespace with all options
        """
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args(args)

        # Set isTrain based on phase
        opt.isTrain = opt.phase == 'train'

        # Parse GPU IDs
        gpu_ids = [int(x) for x in opt.gpu_ids.split(',') if x.strip()]
        opt.gpu_ids = gpu_ids

        # Set device
        if torch.cuda.is_available() and len(gpu_ids) > 0:
            opt.device = torch.device(f'cuda:{gpu_ids[0]}')
        else:
            opt.device = torch.device('cpu')

        # Print options
        self._print_options(opt)

        return opt

    def _print_options(self, opt):
        """Print all options."""
        message = '\n----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            message += f'{k:>25}: {v}\n'
        message += '----------------- End -------------------'
        print(message)

        # Save to file
        if opt.isTrain:
            save_dir = Path(opt.checkpoints_dir) / opt.name
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / 'opt.txt', 'w') as f:
                f.write(message)
