#!/usr/bin/env python
"""
Testing/inference script for Swin Transformer MR-to-CT synthesis.

Computes SSIM and PSNR metrics on test set.

Example usage:
    python test_swin.py \
        --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/cyclegan \
        --name swin_baseline_v1 \
        --epoch 200 \
        --gpu_ids 0
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from options.swin_options import SwinOptions
from data.paired_nifti_dataset import PairedNiftiDataset, create_dataloader
from models.swin_model import SwinModel


def denormalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1] range.

    Args:
        tensor: Torch tensor in [-1, 1] range

    Returns:
        Numpy array in [0, 1] range
    """
    return ((tensor + 1) / 2).clamp(0, 1).cpu().numpy()


def compute_metrics(pred, target):
    """Compute SSIM and PSNR between prediction and target.

    Args:
        pred: Predicted image [H, W] in [0, 1]
        target: Target image [H, W] in [0, 1]

    Returns:
        dict with 'ssim' and 'psnr' values
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    ssim_val = ssim(target, pred, data_range=1.0)
    psnr_val = psnr(target, pred, data_range=1.0)

    return {'ssim': ssim_val, 'psnr': psnr_val}


def main():
    # Parse options
    parser = SwinOptions()
    parser.initialize()

    # Override defaults for testing
    parser.parser.set_defaults(phase='test', batch_size=1)

    opt = parser.parse()
    opt.isTrain = False

    # Create dataset and dataloader
    dataloader = create_dataloader(opt)
    dataset_size = len(dataloader.dataset)
    print(f'Test images: {dataset_size}')

    # Create model and load checkpoint
    model = SwinModel(opt)
    model.load_networks(opt.epoch)
    model.eval()

    # Output directory for results
    results_dir = Path(opt.checkpoints_dir) / opt.name / f'results_{opt.epoch}'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Metrics storage
    all_ssim = []
    all_psnr = []

    print(f'Running inference...')
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            model.set_input(data)
            model.test()

            # Get images
            visuals = model.get_current_visuals()
            real_A = denormalize(visuals['real_A'][0, 0])  # [H, W]
            fake_B = denormalize(visuals['fake_B'][0, 0])  # [H, W]
            real_B = denormalize(visuals['real_B'][0, 0])  # [H, W]

            # Compute metrics
            metrics = compute_metrics(fake_B, real_B)
            all_ssim.append(metrics['ssim'])
            all_psnr.append(metrics['psnr'])

            # Save every 100th image for visualization
            if i % 100 == 0:
                fname = Path(model.image_paths[0]).stem
                save_comparison(
                    real_A, fake_B, real_B,
                    results_dir / f'{fname}_comparison.png'
                )

    # Compute and print final metrics
    mean_ssim = np.mean(all_ssim)
    std_ssim = np.std(all_ssim)
    mean_psnr = np.mean(all_psnr)
    std_psnr = np.std(all_psnr)

    print('\n' + '=' * 50)
    print(f'Results for epoch {opt.epoch}')
    print('=' * 50)
    print(f'SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}')
    print(f'PSNR: {mean_psnr:.2f} +/- {std_psnr:.2f} dB')
    print('=' * 50)

    # Save metrics to file
    with open(results_dir / 'metrics.txt', 'w') as f:
        f.write(f'Epoch: {opt.epoch}\n')
        f.write(f'Test samples: {dataset_size}\n')
        f.write(f'SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}\n')
        f.write(f'PSNR: {mean_psnr:.2f} +/- {std_psnr:.2f} dB\n')


def save_comparison(real_A, fake_B, real_B, save_path):
    """Save side-by-side comparison image.

    Args:
        real_A: MR input [H, W]
        fake_B: Generated CT [H, W]
        real_B: Real CT [H, W]
        save_path: Path to save image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(real_A, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Input MR')
    axes[0].axis('off')

    axes[1].imshow(fake_B, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Generated CT')
    axes[1].axis('off')

    axes[2].imshow(real_B, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Real CT')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
