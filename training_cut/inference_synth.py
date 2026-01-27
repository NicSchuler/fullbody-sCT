#!/usr/bin/env python
"""
Inference script for MR-to-CT synthesis using CUT model.

This is a simplified version of test_synth.py for inference-only mode:
- Uses single dataset mode (MR only, no paired CT)
- Does not compute metrics (no ground truth available)
- Saves fake CT slices as NIfTI files

Usage:
    python inference_synth.py \
        --dataroot /path/to/slices/A \
        --name cut_synthrad_allregions_final \
        --checkpoints_dir /path/to/checkpoints \
        --epoch 50 \
        --results_dir /path/to/output

This script:
    1. Loads MR slices from --dataroot
    2. Runs CUT model inference
    3. Saves fake CT slices to --results_dir/{name}/test_{epoch}/fake_nifti/
"""

import os
import sys
import pathlib
import shutil

# Ensure repository root is on sys.path so imports work
THIS_FILE = pathlib.Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent
for _ in range(8):
    if (REPO_ROOT / 'training_cut' / 'options').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_cut.options.test_options import TestOptions
from training.data import create_dataset
from training_cut.models import create_model
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

# CT value ranges for denormalization (matching test_synth.py)
CT_MAX_VALUE = 1200.0
CT_MIN_VALUE = -1024.0


def save_fake_ct_as_nifti(fake_tensor, img_path, output_dir):
    """
    Convert model output to NIfTI and save.

    Args:
        fake_tensor: Model output tensor (normalized to [-1, 1])
        img_path: Original input path (for filename)
        output_dir: Directory to save NIfTI files
    """
    # Convert from [-1, 1] to HU values
    # Model output is in [-1, 1], we map to [0, 1] then to HU
    fake_norm = fake_tensor[0].cpu().float().numpy().squeeze()
    fake_hu = (fake_norm + 1) / 2 * (CT_MAX_VALUE - CT_MIN_VALUE) + CT_MIN_VALUE
    fake_ct_numpy = np.clip(fake_hu, CT_MIN_VALUE, CT_MAX_VALUE).astype(np.float32)

    # Apply rotation/flip to match expected orientation (same as test_synth.py)
    fake_ct_numpy = np.rot90(fake_ct_numpy, -1)
    fake_ct_numpy = np.fliplr(fake_ct_numpy)

    # Parse filename: "AB_1ABA005-42.nii" -> "AB_1ABA005_42.nii"
    file_name = os.path.basename(img_path[0])
    # Replace hyphen with underscore for volume reconstructor compatibility
    base_name = file_name.replace('-', '_')
    # Remove any double extensions
    if base_name.endswith('.nii.nii'):
        base_name = base_name[:-4]
    if not base_name.endswith('.nii'):
        base_name = os.path.splitext(base_name)[0] + '.nii'

    output_path = os.path.join(output_dir, base_name)

    # Create NIfTI and save
    # Use identity affine with 1mm spacing
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(fake_ct_numpy[:, :, np.newaxis], affine)
    nib.save(nifti_img, output_path)


def main():
    opt = TestOptions().parse()

    # Hard-code inference settings
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1  # no visdom display
    opt.phase = "test"

    # For single (MR-only) dataset mode with NIfTI support
    # The dataroot should point directly to the directory containing MR slices
    opt.dataset_mode = "single_nifti"

    # Set device (required by BaseModel)
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("MR-to-CT Inference (CUT Model)")
    print("=" * 60)
    print(f"Model:           {opt.name}")
    print(f"Epoch:           {opt.epoch}")
    print(f"Data root:       {opt.dataroot}")
    print(f"Checkpoints:     {opt.checkpoints_dir}")
    print(f"Input channels:  {opt.input_nc}")
    print(f"Output channels: {opt.output_nc}")
    print("=" * 60)

    # Create output directory
    output_dir = os.path.join(
        opt.results_dir,
        opt.name,
        f"test_{opt.epoch}",
        "fake_nifti"
    )

    # Clean existing output if present
    if os.path.exists(output_dir):
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir:      {output_dir}")
    print()

    # Create dataset and model
    dataset = create_dataset(opt)
    model = create_model(opt)

    # Setup model (loads weights)
    model.setup(opt)

    # Set to eval mode
    if opt.eval:
        model.eval()

    print(f"Processing {len(dataset)} images...")
    print()

    # Process all images
    count = 0
    for i, data in enumerate(tqdm(dataset, desc="Inference")):
        # CUT model expects 'B' key even during inference (not used, but required by set_input)
        if 'B' not in data:
            data['B'] = data['A']  # dummy B for inference
        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        # Save fake CT
        if "fake_B" in visuals:
            save_fake_ct_as_nifti(visuals["fake_B"], img_path, output_dir)
        elif "fake" in visuals:
            save_fake_ct_as_nifti(visuals["fake"], img_path, output_dir)
        else:
            # Try to find any fake output
            for key in visuals:
                if "fake" in key.lower():
                    save_fake_ct_as_nifti(visuals[key], img_path, output_dir)
                    break

        count += 1

        # Progress update
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...")

        # Limit number of test images if specified
        if opt.num_test and i >= opt.num_test - 1:
            break

    print()
    print("=" * 60)
    print(f"Inference complete!")
    print(f"  Processed: {count} slices")
    print(f"  Output:    {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
