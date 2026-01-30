#!/usr/bin/env python
"""
Load input data from Grand Challenge format to internal pipeline format.

Converts GC input structure to 1initNifti with region-based patient IDs.

Input structure (GC format):
    gc_input/
    ├── region.json              # {"region": "Head and Neck"|"Abdomen"|"Thorax"}
    └── images/
        ├── mri/                  # MRI files (.nii.gz, .nii, .mha)
        │   └── *.nii.gz
        └── body/                 # Optional body masks
            └── *.nii.gz

Output structure:
    init_dir/
    ├── input_mapping.json        # {patient_id: original_filename}
    ├── region_prefix.txt         # Region prefix (HN, AB, TH)
    └── {patient_id}/
        ├── MR/
        │   └── {patient_id}_MR.nii.gz
        └── masks/
            └── {patient_id}_mask.nii.gz
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import SimpleITK as sitk


REGION_PREFIX_MAP = {
    "Head and Neck": "HN",
    "Abdomen": "AB",
    "Thorax": "TH",
}


def convert_to_nii_gz(src: str, dst: str) -> None:
    """Copy if already .nii.gz, otherwise convert via SimpleITK."""
    if src.endswith(".nii.gz"):
        shutil.copy2(src, dst)
    else:
        img = sitk.ReadImage(src)
        sitk.WriteImage(img, dst)


def discover_image_files(directory: str) -> list[str]:
    """Find all supported image files in a directory."""
    if not os.path.isdir(directory):
        return []

    return sorted(
        glob.glob(os.path.join(directory, "*.nii.gz"))
        + glob.glob(os.path.join(directory, "*.nii"))
        + glob.glob(os.path.join(directory, "*.mha"))
    )


def load_input(gc_input: str, init_dir: str) -> None:
    """Convert GC input format to internal pipeline format."""

    # Read region from region.json
    region_json_path = os.path.join(gc_input, "region.json")
    if not os.path.isfile(region_json_path):
        print(f"ERROR: region.json not found: {region_json_path}")
        sys.exit(1)

    with open(region_json_path) as f:
        region = json.load(f)

    # Get region prefix
    prefix = REGION_PREFIX_MAP.get(region)
    if not prefix:
        print(f"ERROR: Unknown region: {region}")
        sys.exit(1)

    print(f"Region: {region} -> Prefix: {prefix}")

    # Discover MRI files
    mri_dir = os.path.join(gc_input, "images", "mri")
    mri_files = discover_image_files(mri_dir)

    if not mri_files:
        print(f"ERROR: No image files found in {mri_dir}")
        sys.exit(1)

    # Discover body mask files (optional)
    body_dir = os.path.join(gc_input, "images", "body")
    body_files = discover_image_files(body_dir)

    # Create output directory
    os.makedirs(init_dir, exist_ok=True)

    # Mapping: patient_id -> original MRI filename (for provide_output.py)
    input_mapping = {}

    for idx, mri_path in enumerate(mri_files):
        basename = os.path.basename(mri_path)
        patient_id = f"{prefix}_{idx + 1:03d}"

        input_mapping[patient_id] = basename

        # Create directory structure
        mr_dir = os.path.join(init_dir, patient_id, "MR")
        mask_dir = os.path.join(init_dir, patient_id, "masks")
        os.makedirs(mr_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # Copy/convert MRI to .nii.gz
        dst_mri = os.path.join(mr_dir, f"{patient_id}_MR.nii.gz")
        convert_to_nii_gz(mri_path, dst_mri)
        print(f"  MRI:  {mri_path} -> {dst_mri}")

        # Copy/convert body mask if available (matched by sorted index)
        if idx < len(body_files):
            dst_body = os.path.join(mask_dir, f"{patient_id}_mask.nii.gz")
            convert_to_nii_gz(body_files[idx], dst_body)
            print(f"  Mask: {body_files[idx]} -> {dst_body}")

    # Save mapping for provide_output.py
    mapping_path = os.path.join(init_dir, "input_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(input_mapping, f)
    print(f"  Mapping saved to {mapping_path}")

    # Save prefix for model name determination
    prefix_path = os.path.join(init_dir, "region_prefix.txt")
    with open(prefix_path, "w") as f:
        f.write(prefix)
    print(f"  Region prefix saved to {prefix_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load input data from Grand Challenge format to internal pipeline format."
    )
    parser.add_argument(
        "--gc-input",
        required=True,
        help="Grand Challenge input directory",
    )
    parser.add_argument(
        "--init-dir",
        required=True,
        help="Output directory for converted files (1initNifti)",
    )

    args = parser.parse_args()

    load_input(args.gc_input, args.init_dir)


if __name__ == "__main__":
    main()
