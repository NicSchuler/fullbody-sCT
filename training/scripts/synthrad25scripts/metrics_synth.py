# some metrics based on implementation https://aramislab.paris.inria.fr/workshops/DL4MI/2021/notebooks/GAN.html
import torch
import numpy as np
import os
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as peak_signal_noise_ratio


def mean_absolute_error(image_true, image_generated,slice_mask): # , arr_diff_numerical):
    """Compute mean absolute error.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image

    """
    diff = np.abs(image_true - image_generated)

    if slice_mask is None or slice_mask.shape != diff.shape:
        diff_masked = diff
    else:
        # works for both boolean masks and 0/1 masks
        diff_masked = diff[slice_mask != 0]
    mae = diff_masked.mean()
    return mae  # , arr_diff_numerical


def mean_squared_error(image_true, image_generated,slice_mask):
    diff = abs(image_true - image_generated)
    if slice_mask is None or slice_mask.shape!=diff.shape: 
        diff_masked = diff
    else:
        diff_masked = diff[slice_mask ==1]

    mse = (diff_masked ** 2).mean()
    return mse


def peak_signal_to_noise_ratio(image_true, image_generated,slice_mask,data_range=2224.0):
    """"Compute peak signal-to-noise ratio.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image

    Returns:
        psnr: (float) peak signal-to-noise ratio"""
    if slice_mask is not None:
        true = image_true[slice_mask == 1]
        fake = image_generated[slice_mask == 1]
    else:
        true = image_true
        fake = image_generated

    psnr = peak_signal_noise_ratio(
        true,
        fake,
        data_range=data_range,
    )
    return float(psnr)


def structural_similarity_index_skimage(image_true, image_generated, slice_mask=None, data_range=1.0):
    """Compute structural similarity index using skimage, optionally masked."""
    score, ssim_map = ssim(image_true, image_generated, data_range=data_range, full=True)
    if slice_mask is None or slice_mask.shape != ssim_map.shape:
        return float(score), float(ssim_map.mean())
    masked_score = ssim_map[slice_mask != 0].mean()
    return float(score), float(masked_score)
