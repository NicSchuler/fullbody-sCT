# some metrics based on implementation https://aramislab.paris.inria.fr/workshops/DL4MI/2021/notebooks/GAN.html
import torch
import numpy as np
import os
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as peak_signal_noise_ratio


def _valid_mask(slice_mask, shape):
    if slice_mask is None:
        return None
    if slice_mask.shape != shape:
        return None
    mask = slice_mask != 0
    if mask.sum() == 0:
        return None
    return mask


def mean_absolute_error(image_true, image_generated, slice_mask=None):  # , arr_diff_numerical):
    """Compute mean absolute error.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image

    """
    diff = np.abs(image_true - image_generated)
    mae_unmasked = diff.mean()
    mask = _valid_mask(slice_mask, diff.shape)
    if mask is None:
        mae_masked = mae_unmasked
    else:
        mae_masked = diff[mask].mean()
    return float(mae_unmasked), float(mae_masked)  # , arr_diff_numerical


def mean_squared_error(image_true, image_generated, slice_mask=None):
    diff = abs(image_true - image_generated)
    mse_unmasked = (diff ** 2).mean()
    mask = _valid_mask(slice_mask, diff.shape)
    if mask is None:
        mse_masked = mse_unmasked
    else:
        mse_masked = (diff[mask] ** 2).mean()
    return float(mse_unmasked), float(mse_masked)


def peak_signal_to_noise_ratio(image_true, image_generated, slice_mask=None, data_range=2224.0):
    """"Compute peak signal-to-noise ratio.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image

    Returns:
        psnr: (float) peak signal-to-noise ratio"""
    psnr_unmasked = peak_signal_noise_ratio(
        image_true,
        image_generated,
        data_range=data_range,
    )
    mask = _valid_mask(slice_mask, image_true.shape)
    if mask is None:
        psnr_masked = psnr_unmasked
    else:
        psnr_masked = peak_signal_noise_ratio(
            image_true[mask],
            image_generated[mask],
            data_range=data_range,
        )
    return float(psnr_unmasked), float(psnr_masked)


def structural_similarity_index_skimage(image_true, image_generated, slice_mask=None, data_range=1.0):
    """Compute structural similarity index using skimage, optionally masked."""
    score, ssim_map = ssim(image_true, image_generated, data_range=data_range, full=True)
    mask = _valid_mask(slice_mask, ssim_map.shape)
    if mask is None:
        masked_score = score
    else:
        masked_score = ssim_map[mask].mean()
    return float(score), float(masked_score)
