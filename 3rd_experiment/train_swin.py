#!/usr/bin/env python
"""
Training script for Swin Transformer MR-to-CT synthesis.

Example usage:
    python train_swin.py \
        --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/cyclegan \
        --name swin_baseline_v1 \
        --batch_size 8 \
        --n_epochs 200 \
        --lr 1e-4 \
        --gpu_ids 0
"""

import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from options.swin_options import SwinOptions
from data.paired_nifti_dataset import PairedNiftiDataset, create_dataloader
from models.swin_model import SwinModel


def main():
    # Parse options
    opt = SwinOptions().parse()

    # Create dataset and dataloader
    dataloader = create_dataloader(opt)
    dataset_size = len(dataloader.dataset)
    print(f'Training images: {dataset_size}')

    # Create model
    model = SwinModel(opt)

    # Setup tensorboard
    log_dir = Path(opt.checkpoints_dir) / opt.name / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Resume training if specified
    start_epoch = 1
    if opt.continue_train:
        model.load_networks(opt.epoch)
        if opt.epoch != 'latest':
            start_epoch = int(opt.epoch) + 1
        print(f'Resuming from epoch {start_epoch}')

    # Training loop
    total_iters = 0
    for epoch in range(start_epoch, opt.n_epochs + 1):
        epoch_start = time.time()
        epoch_iter = 0
        running_loss = 0.0

        model.train()
        for i, data in enumerate(dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # Forward and backward
            model.set_input(data)
            model.optimize_parameters()

            # Accumulate loss
            losses = model.get_current_losses()
            running_loss += losses['L1']

            # Print progress
            if total_iters % opt.print_freq == 0:
                current_lr = model.optimizer_G.param_groups[0]['lr']
                print(f'Epoch [{epoch}/{opt.n_epochs}] '
                      f'Iter [{epoch_iter}/{dataset_size}] '
                      f'L1: {losses["L1"]:.4f} '
                      f'LR: {current_lr:.2e}')

        # Epoch statistics
        avg_loss = running_loss / len(dataloader)
        epoch_time = time.time() - epoch_start

        # Log to tensorboard
        writer.add_scalar('Loss/L1', avg_loss, epoch)
        writer.add_scalar('LR', model.optimizer_G.param_groups[0]['lr'], epoch)

        # Log sample images every 10 epochs
        if epoch % 10 == 0:
            visuals = model.get_current_visuals()
            for name, img in visuals.items():
                # Denormalize from [-1,1] to [0,1]
                img = (img + 1) / 2
                writer.add_images(f'Images/{name}', img, epoch)

        print(f'Epoch {epoch} completed in {epoch_time:.1f}s | Avg L1: {avg_loss:.4f}')

        # Update learning rate
        model.update_learning_rate()

        # Save checkpoint
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)

        # Always save latest
        model.save_networks('latest')

    # Save final model
    model.save_networks('final')
    writer.close()
    print('Training complete!')


if __name__ == '__main__':
    main()
