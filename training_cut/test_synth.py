"""General-purpose test script for image-to-image translation.

Copied from USZ Pipeline repo and modified for SynthRad25 project

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import sys
import pathlib

# Ensure repository root is on sys.path so 'training' imports work
THIS_FILE = pathlib.Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent
for _ in range(8):
    if (REPO_ROOT / 'training_cut' / 'options').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.scripts.synthrad25scripts.metrics_synth import (
    structural_similarity_index_skimage,
    peak_signal_to_noise_ratio,
    mean_absolute_error,
    mean_squared_error,
)
from training_cut.options.test_options import TestOptions
from training.data import create_dataset
from training_cut.models import create_model
from training.util.visualizer import save_images
from training.util import html
import torch
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
import shutil
from pathlib import Path
from tqdm import tqdm 
import util.util as util
# from pytorch_fid import fid_score

BASE_ROOT = os.environ.get(
    "SYNTHRAD_BASE_ROOT",
    "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed",
)

# ---- body mask config ----
use_mask = True  # set True to use masks when computing metrics
mask_slice_base_dir = os.environ.get(
    "SYNTHRAD_MASK_SLICE_DIR",
    os.path.join(BASE_ROOT, "5slices_31baseline", "masks"),
)

ct_max_value=1200.0
ct_min_value=-1024.0

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.device = 'cuda'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    data_root = Path(opt.dataroot)
    if (opt.model == 'pix2pix'):
        ct_slice_dir = data_root.parent / "test" / "B"
    else:
        ct_slice_dir = data_root.parent / "test" / "testB"
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # prepare metrics
    fake_key = 'fake_' + opt.direction[-1]
    real_key = 'real_' + opt.direction[-1]

    res_test = []
    skipped_zero_mask_rows = 0

    results_path = os.path.join(
        BASE_ROOT,
        "9latestTestImages",
        opt.name,
        '{}_{}'.format(opt.phase, opt.epoch),
    )

    if os.path.exists(results_path):
        print("Such path {} exists. Removing it".format(results_path))
        try:
            shutil.rmtree(results_path)
        except OSError as e:
            print("Error: %s : %s" % (results_path, e.strerror))

    os.makedirs(results_path)


    model.setup(opt)
    if opt.eval:
        model.eval()


    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        #model and load image
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results real_a, fake_B, real_B
        img_path = model.get_image_paths()     # get image paths

        #fake image processing
        fake_norm = visuals["fake_B"][0].cpu().float().numpy().squeeze()
        fake_hu = (fake_norm+1)/2 * (ct_max_value - ct_min_value) + ct_min_value
        fake_ct_numpy = np.clip(fake_hu, ct_min_value, ct_max_value).astype(np.int16)
        fake_ct_numpy = np.rot90(fake_ct_numpy, -1)
        fake_ct_numpy = np.fliplr(fake_ct_numpy)

        #real image processing
        file_name = os.path.basename(img_path[0])
        treatment,slice = file_name.split('-')
        slice= slice.split('.')[0]
        slice_idx = file_name.split('-')[1].split('.')[0]
        real_ct_path = os.path.join(ct_slice_dir, file_name)
        real_ct_image = nib.load(real_ct_path)
        real_ct_nii_array = real_ct_image.get_fdata().squeeze()  # normalized [0,1]

        real_hu = real_ct_nii_array * (ct_max_value - ct_min_value) + ct_min_value
        real_ct_numpy = np.clip(real_hu, ct_min_value, ct_max_value).astype(np.int16)
        real_ct_numpy = np.rot90(real_ct_numpy, -1)
        real_ct_numpy = np.fliplr(real_ct_numpy)

        mask = None
        mask_voxels = 0
        if use_mask:
            #load mask
            mask_slice_path = os.path.join(mask_slice_base_dir, file_name)
            if os.path.exists(mask_slice_path):
                mask_img = nib.load(mask_slice_path)
                mask_array = mask_img.get_fdata().squeeze()

                # apply same transforms as for CT images
                mask_array = np.rot90(mask_array, -1)
                mask_array = np.fliplr(mask_array)
                mask = mask_array.astype(bool)
                mask_voxels = int(mask.sum())
            else:
                print(f"Warning: Mask not found for {file_name}, skipping mask.")

        # compute metrics
        mae_unmasked, mae_masked = mean_absolute_error(real_ct_numpy, fake_ct_numpy, mask)
        mse_unmasked, mse_masked = mean_squared_error(real_ct_numpy, fake_ct_numpy, mask)
        psnr_unmasked, psnr_masked = peak_signal_to_noise_ratio(
            real_ct_numpy,
            fake_ct_numpy,
            mask,
            data_range=ct_max_value - ct_min_value,
        )
        fake_norm_01 = (fake_norm + 1) / 2
        ssim_unmasked, ssim_masked = structural_similarity_index_skimage(
            real_ct_nii_array,
            fake_norm_01,
            mask,
            data_range=1.0,
        )
        # Keep only slices with non-empty masks in the metrics CSV.
        if mask_voxels > 0:
            res_test.append([
                file_name,
                mae_unmasked,
                mae_masked,
                mse_unmasked,
                mse_masked,
                psnr_unmasked,
                psnr_masked,
                ssim_unmasked,
                ssim_masked,
                mask_voxels,
            ])
        else:
            skipped_zero_mask_rows += 1


        #save dicoms and niftis
        # ----- save NIfTI for visualization -----
        # fake
        path_fake_nifti = os.path.join(results_path, "fake_nifti")
        os.makedirs(path_fake_nifti, exist_ok=True)
        fake_ct_nif = nib.Nifti1Image(fake_ct_numpy, np.eye(4))
        nib.save(fake_ct_nif, os.path.join(path_fake_nifti, treatment + "_" + slice_idx + ".nii"))

        # real
        path_real_nifti = os.path.join(results_path, "real_nifti")
        os.makedirs(path_real_nifti, exist_ok=True)
        real_ct_nif = nib.Nifti1Image(real_ct_numpy, np.eye(4))
        nib.save(real_ct_nif, os.path.join(path_real_nifti, treatment + "_" + slice_idx + ".nii"))


        if i % 1000 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        if opt.save_results:
            save_images(webpage, visuals, img_path, width=opt.display_winsize)
            webpage.save()  # save the HTML


    print("Results for test split, mean:")
    metric_columns = [
        "MAE_unmasked", "MAE_masked",
        "MSE_unmasked", "MSE_masked",
        "PSNR_unmasked", "PSNR_masked",
        "SSIM_unmasked", "SSIM_masked",
    ]
    output_columns = ["file_name", *metric_columns, "mask_voxels"]
    metrics_df = pd.DataFrame(res_test, columns=output_columns)
    print(f"Skipped {skipped_zero_mask_rows} slices with mask_voxels == 0 when writing CSV metrics.")
    df = pd.DataFrame([metrics_df[metric_columns].mean().squeeze()], index=['Test set']).T

    metrics_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch}")
    metrics_df.to_csv(
        os.path.join(metrics_dir, "test_metrics_over_all.csv")
    )

    print(df)

    print("Results for test split, standard deviation:")
    st_d_df = pd.DataFrame([metrics_df[metric_columns].std().squeeze()], index=['Test set']).T

    st_d_df.to_csv(os.path.join(metrics_dir, "test_metrics_std_over_all.csv"))

    print(st_d_df)
