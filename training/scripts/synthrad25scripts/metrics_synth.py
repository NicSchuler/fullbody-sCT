# some metrics based on implementation https://aramislab.paris.inria.fr/workshops/DL4MI/2021/notebooks/GAN.html
import torch
import numpy as np
import os
import nibabel as nib
from skimage.metrics import structural_similarity as ssim


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

def mean_and_graph_absolute_error(real_dct_numpy, fake_ct_numpy, slice_mask_air_or, air_ct_value, results_path,treatment,slice, sCT_diff, dCT_diff):

    """Compute mean absolute error.

    Args:
        real_dct_numpy: (Tensor) true dCT image
        fake_ct_numpy: (Tensor) generated image
        slice_mask_air_or: (Tensor) mask of pre-contoured air bubbles

    """
    slice_mask_air_or=slice_mask_air_or*air_ct_value
    diff_to_sCT=abs(fake_ct_numpy - slice_mask_air_or)
    diff_to_dCT = abs(real_dct_numpy - slice_mask_air_or)
    mae_air_or_sCT=0
    mae_air_or_dCT=0


    if slice_mask_air_or.shape==diff_to_sCT.shape and slice_mask_air_or.shape==diff_to_dCT.shape:
        diff_to_sCT_masked = diff_to_sCT[slice_mask_air_or ==air_ct_value]
        diff_to_dCT_masked = diff_to_dCT[slice_mask_air_or == air_ct_value]

        ##check where the diff is 0 or no air masks#
        mae_air_or_sCT=(np.abs(diff_to_sCT_masked)).mean()
        mae_air_or_dCT=(np.abs(diff_to_dCT_masked)).mean()


        sCT_to_merge=diff_to_sCT_masked.flatten()
        dCT_to_merge = diff_to_dCT_masked.flatten()

        sCT_diff.append(sCT_to_merge)
        dCT_diff.append(dCT_to_merge)

    else:
        print("mask_error!")
    return mae_air_or_sCT,mae_air_or_dCT, sCT_diff, dCT_diff



#https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/simple_metrics.py

def mean_squared_error(image_true, image_generated,slice_mask):
    diff = abs(image_true - image_generated)
    if slice_mask is None or slice_mask.shape!=diff.shape: 
        diff_masked = diff
    else:
        diff_masked = diff[slice_mask ==1]

    mse = (diff_masked ** 2).mean()
    return mse


def peak_signal_to_noise_ratio(image_true, image_generated,slice_mask):
    """"Compute peak signal-to-noise ratio.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image

    Returns:
        psnr: (float) peak signal-to-noise ratio"""
    diff = abs(image_true - image_generated)

    if slice_mask is None or slice_mask.shape!=diff.shape: 
        diff_masked = diff
    else:
        diff_masked = diff[slice_mask ==1]
    mse = (diff_masked ** 2).mean()

    max_pixel = 3071+1024
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def structural_similarity_index_skimage(image_true, image_generated, slice_mask=None, data_range=1.0):
    """Compute structural similarity index using skimage, optionally masked."""
    score, ssim_map = ssim(image_true, image_generated, data_range=data_range, full=True)
    if slice_mask is None or slice_mask.shape != ssim_map.shape:
        return float(score), float(ssim_map.mean())
    masked_score = ssim_map[slice_mask != 0].mean()
    return float(score), float(masked_score)
