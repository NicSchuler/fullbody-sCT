#!/usr/bin/env python
from __future__ import annotations
"""
Copy results from internal pipeline format to Grand Challenge output format.

Converts reconstructed sCT files back to the original input format and
generates the results.json file for Grand Challenge.

Input structure:
    reconstruction_dir/
    └── {patient_id}/
        └── sCT_original_dim_reconstructed_alignment.nii.gz

    init_dir/
    └── input_mapping.json        # {patient_id: original_filename}

Output structure (GC format):
    gc_output/
    ├── results.json
    └── images/
        └── synthetic-ct/
            └── {original_filename}  # Preserves original format (.mha, .nii, .nii.gz)
"""

import argparse
import json
import os
import shutil
import sys

import SimpleITK as sitk


# NIfTI-specific metadata keys that MetaImage format does not support
NIFTI_METADATA_KEYS = ["ITK_FileNotes", "aux_file", "descrip", "intent_name"]


def provide_output(
    gc_input: str,
    gc_output: str,
    reconstruction_dir: str,
    init_dir: str,
) -> None:
    """Copy reconstructed results to GC output format."""

    # Read input mapping (patient_id -> original filename) saved by load_input.py
    mapping_path = os.path.join(init_dir, "input_mapping.json")
    if not os.path.isfile(mapping_path):
        print(f"ERROR: Input mapping not found: {mapping_path}")
        sys.exit(1)

    with open(mapping_path) as f:
        input_mapping = json.load(f)

    # Create output directory
    output_dir = os.path.join(gc_output, "images", "synthetic-ct")
    os.makedirs(output_dir, exist_ok=True)

    case_results = []

    for patient_id, original_basename in sorted(input_mapping.items()):
        src = os.path.join(
            reconstruction_dir,
            patient_id,
            "sCT_original_dim_reconstructed_alignment.nii.gz",
        )
        if not os.path.exists(src):
            print(f"ERROR: Reconstructed file not found: {src}")
            sys.exit(1)

        # Output uses the original filename (preserving .mha / .nii / .nii.gz)
        output_path = os.path.join(output_dir, original_basename)

        if original_basename.endswith(".nii.gz"):
            shutil.copy2(src, output_path)
        else:
            # Convert from .nii.gz back to original format (.mha, .nii)
            img = sitk.ReadImage(src)
            # Strip NIfTI-specific metadata that MetaImage format does not support
            for key in NIFTI_METADATA_KEYS:
                if img.HasMetaDataKey(key):
                    img.EraseMetaData(key)
            sitk.WriteImage(img, output_path)

        print(f"  {src} -> {output_path}")

        case_results.append(
            {
                "outputs": [{"type": "metaio_image", "filename": output_path}],
                "inputs": [
                    {
                        "type": "metaio_image",
                        "filename": os.path.join(
                            gc_input, "images", "mri", original_basename
                        ),
                    },
                ],
                "error_messages": [],
            }
        )

    # Write results.json
    results_path = os.path.join(gc_output, "results.json")
    with open(results_path, "w") as f:
        json.dump(case_results, f)
    print(f"  Results written to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy results from internal pipeline format to Grand Challenge output format."
    )
    parser.add_argument(
        "--gc-input",
        required=True,
        help="Grand Challenge input directory",
    )
    parser.add_argument(
        "--gc-output",
        required=True,
        help="Grand Challenge output directory",
    )
    parser.add_argument(
        "--reconstruction-dir",
        required=True,
        help="Directory containing reconstructed sCT files (10reconstruction)",
    )
    parser.add_argument(
        "--init-dir",
        required=True,
        help="Directory containing input_mapping.json (1initNifti)",
    )

    args = parser.parse_args()

    provide_output(
        args.gc_input,
        args.gc_output,
        args.reconstruction_dir,
        args.init_dir,
    )


if __name__ == "__main__":
    main()
