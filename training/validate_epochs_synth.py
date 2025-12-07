"""Fast validation script over multiple epochs for SynthRad25.

Usage examples:

# Validate a single epoch on full val set
CUDA_VISIBLE_DEVICES=7 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splitsNonNormalizedBodyRegion/AB/pix2pix/AB \
  --name pix2pix_synthrad_abdomen \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50

  # Validate a single epoch on subset val set
CUDA_VISIBLE_DEVICES=7 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/pix2pix/AB \
  --name pix2pix_synthrad_pelvis \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50 \
  --num_test 50

# Validate ALL saved epochs on a random subset of 100 val images
CUDA_VISIBLE_DEVICES=7 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splitsNonNormalizedBodyRegion/AB/pix2pix/AB \
  --name pix2pix_synthrad_abdomen \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch all \
  --num_test 100
"""

import os
import sys
import pathlib
import re
import random

# Ensure repository root is on sys.path so 'training' imports work
THIS_FILE = pathlib.Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent
for _ in range(8):
    if (REPO_ROOT / 'training' / 'options').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.scripts.synthrad25scripts.metrics_synth import (
    structural_similarity_index,
    peak_signal_to_noise_ratio,
    mean_absolute_error,
    mean_squared_error,
)
from training.options.test_options import TestOptions
from training.data import create_dataset
from training.models import create_model
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ---- body mask config ----
use_mask = True  # set True to use masks when computing metrics
mask_slice_base_dir = os.path.join(
    "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/5slices_31baseline/masks"
)

ct_max_value = 1200.0
ct_min_value = -1024.0


if __name__ == "__main__":
    opt = TestOptions().parse()  # get options from CLI

    # ---- Force validation phase ----
    if opt.phase != "val":
        print(f"Overriding phase '{opt.phase}' -> 'val' for validation.")
    opt.phase = "val"

    # ---- Determine which epochs to evaluate ----
    exp_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if opt.epoch == "all":
        print(f"Scanning checkpoints in: {exp_dir}")
        epoch_numbers = set()
        if os.path.isdir(exp_dir):
            for fname in os.listdir(exp_dir):
                m = re.match(r"(\d+)_net_.*\.pth", fname)
                if m:
                    epoch_numbers.add(int(m.group(1)))
        epochs_to_test = sorted(epoch_numbers)
        if not epochs_to_test:
            raise RuntimeError(f"No numeric checkpoints found in {exp_dir}")
        print(f"Will validate epochs: {epochs_to_test}")
        multi_epoch_mode = True
    else:
        epochs_to_test = [opt.epoch]
        multi_epoch_mode = False
        print(f"Will validate single epoch: {opt.epoch}")

    # ---- Hard-code some parameters for test-like behavior ----
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True
    opt.device = "cuda"

    # ---- Dataset & model ----
    dataset = create_dataset(opt)
    model = create_model(opt)

    data_root = Path(opt.dataroot)
    # For pix2pix: .../AB/pix2pix/AB, we expect ../val/B
    # For others:  ../val/valB
    if opt.model == "pix2pix":
        ct_slice_dir = data_root.parent / opt.phase / "B"
    else:
        ct_slice_dir = data_root.parent / opt.phase / f"{opt.phase}B"

    print(f"Using real CT slices from: {ct_slice_dir}")

    # ---- Random subset (using opt.num_test as max size) ----
    dataset_len = len(dataset)
    if getattr(opt, "num_test", 0) and opt.num_test > 0:
        num_subset = min(opt.num_test, dataset_len)
    else:
        num_subset = dataset_len  # full val set if num_test not set

    all_indices = list(range(dataset_len))
    subset_indices = sorted(random.sample(all_indices, num_subset))
    subset_index_set = set(subset_indices)

    print(
        f"Validation on random subset of {num_subset}/{dataset_len} images "
        f"(same subset for all epochs)."
    )

    # ---- Where to store summary CSV ----
    summary_dir = os.path.join(opt.results_dir, opt.name, "val_epoch_metrics")
    os.makedirs(summary_dir, exist_ok=True)
    summary_rows = []

    # ---------------- LOOP OVER EPOCHS ----------------
    for epoch in epochs_to_test:
        print("\n" + "=" * 60)
        print(f"Evaluating epoch: {epoch}")
        print("=" * 60)

        opt.epoch = str(epoch)

        # (Re)load weights for this epoch
        model.setup(opt)
        if opt.eval:
            model.eval()

        res = []

        # Iterate over dataset but only process random subset indices
        for i, data in tqdm(enumerate(dataset), total=len(dataset)):
            if i not in subset_index_set:
                continue

            # Forward pass
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()

            # ---- fake image processing ----
            fake_norm = visuals["fake_B"][0].cpu().float().numpy().squeeze()
            fake_hu = (fake_norm + 1) / 2 * (ct_max_value - ct_min_value) + ct_min_value
            fake_ct_numpy = np.clip(fake_hu, ct_min_value, ct_max_value).astype(np.int16)
            fake_ct_numpy = np.rot90(fake_ct_numpy, -1)
            fake_ct_numpy = np.fliplr(fake_ct_numpy)

            # ---- real image processing ----
            file_name = os.path.basename(img_path[0])
            treatment, slice_ = file_name.split("-")
            slice_ = slice_.split(".")[0]
            slice_idx = file_name.split("-")[1].split(".")[0]

            real_ct_path = os.path.join(ct_slice_dir, file_name)
            real_ct_image = nib.load(real_ct_path)
            real_ct_nii_array = real_ct_image.get_fdata().squeeze()  # normalized [0,1]

            real_hu = real_ct_nii_array * (ct_max_value - ct_min_value) + ct_min_value
            real_ct_numpy = np.clip(real_hu, ct_min_value, ct_max_value).astype(np.int16)
            real_ct_numpy = np.rot90(real_ct_numpy, -1)
            real_ct_numpy = np.fliplr(real_ct_numpy)

            # ---- mask (optional) ----
            mask = None
            if use_mask:
                mask_slice_path = os.path.join(mask_slice_base_dir, file_name)
                if os.path.exists(mask_slice_path):
                    mask_img = nib.load(mask_slice_path)
                    mask_array = mask_img.get_fdata().squeeze()
                    mask_array = np.rot90(mask_array, -1)
                    mask_array = np.fliplr(mask_array)
                    mask = mask_array.astype(bool)
                else:
                    print(f"Warning: Mask not found for {file_name}, skipping mask.")

            # ---- metrics ----
            mae = mean_absolute_error(real_ct_numpy, fake_ct_numpy, mask)
            mse = mean_squared_error(real_ct_numpy, fake_ct_numpy, mask)
            psnr = peak_signal_to_noise_ratio(real_ct_numpy, fake_ct_numpy, mask)
            ssim = structural_similarity_index(real_ct_numpy, fake_ct_numpy)

            res.append([mae, mse, psnr, ssim])

        # ---- metrics for this epoch ----
        res_df = pd.DataFrame(res, columns=["MAE", "MSE", "PSNR", "SSIM"])

        df_mean = res_df.mean().to_frame(name="mean")
        df_std = res_df.std().to_frame(name="std")

        print(f"\nResults for validation subset (epoch {opt.epoch}) - mean:")
        print(df_mean)
        print(f"\nResults for validation subset (epoch {opt.epoch}) - std:")
        print(df_std)

        # Add to summary (one row per epoch: means)
        row = df_mean["mean"].to_dict()
        row["epoch"] = int(epoch)
        summary_rows.append(row)

        # Optional: per-epoch CSVs (comment out if you don't want them)
        df_mean.to_csv(
            os.path.join(summary_dir, f"metrics_mean_epoch_{opt.epoch}.csv")
        )
        df_std.to_csv(
            os.path.join(summary_dir, f"metrics_std_epoch_{opt.epoch}.csv")
        )

    # ---- final summary across epochs ----
    summary_df = pd.DataFrame(summary_rows).set_index("epoch").sort_index()
    print("\n===== Summary over epochs (validation subset, means) =====")
    print(summary_df)

    summary_csv_path = os.path.join(summary_dir, "metrics_summary_over_epochs.csv")
    summary_df.to_csv(summary_csv_path)
    print(f"\nSaved epoch summary to: {summary_csv_path}")
