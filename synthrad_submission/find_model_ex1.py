#!/usr/bin/env python
from __future__ import annotations
"""
Determine the model name based on model type, body region type, and region prefix.

Reads the region prefix from the init directory and outputs the corresponding
model name. Optionally validates that the checkpoint file exists.

Usage:
    python find_model_ex1.py --model-type CUT --bodyregion-type allregions \
        --init-dir /path/to/1initNifti --checkpoint-dir /path/to/checkpoints --epoch 50
"""

import argparse
import os
import sys


# Model name lookup table: model_type -> region_key -> model_name
MODEL_NAMES = {
    "CUT": {
        "AB": "cut_synthrad_abdomen_final",
        "HN": "cut_synthrad_HN_final",
        "TH": "cut_synthrad_TH_final",
        "allregions": "cut_synthrad_allregions_final",
    },
    "cycleGAN": {
        "AB": "cyclegan_abdomen_final",
        "HN": "cyclegan_head_neck_final",
        "TH": "cyclegan_thorax_final",
        "allregions": "cyclegan_allregions_final",
    },
    "pix2pix": {
        "AB": "pix2pix_synthrad_abdomen_final",
        "HN": "pix2pix_synthrad_headneck_final",
        "TH": "pix2pix_synthrad_thorax_final",
        "allregions": "pix2pix_synthrad_allregion_final",
    },
}


def get_model_name(
    model_type: str,
    bodyregion_type: str,
    region_prefix: str,
) -> str:
    """Look up the model name based on model type and region."""
    # Determine the region key
    if bodyregion_type == "allregions":
        region_key = "allregions"
    elif bodyregion_type == "regionspecific":
        region_key = region_prefix
    else:
        print(f"ERROR: Unknown bodyregion_type: {bodyregion_type}", file=sys.stderr)
        sys.exit(1)

    # Look up model type
    if model_type not in MODEL_NAMES:
        print(f"ERROR: Unknown model_type: {model_type}", file=sys.stderr)
        sys.exit(1)

    # Look up region key
    region_map = MODEL_NAMES[model_type]
    if region_key not in region_map:
        print(f"ERROR: Unknown region key: {region_key}", file=sys.stderr)
        sys.exit(1)

    return region_map[region_key]


def find_model(
    model_type: str,
    bodyregion_type: str,
    init_dir: str,
    checkpoint_dir: str,
    epoch: str,
) -> None:
    """Determine model name and validate checkpoint."""
    # Read the region prefix from Step 0
    prefix_path = os.path.join(init_dir, "region_prefix.txt")
    if not os.path.isfile(prefix_path):
        print(f"ERROR: Region prefix file not found: {prefix_path}", file=sys.stderr)
        sys.exit(1)

    with open(prefix_path) as f:
        region_prefix = f.read().strip()

    # Get model name
    model_name = get_model_name(model_type, bodyregion_type, region_prefix)

    print(f"Model name determined: {model_name}", file=sys.stderr)
    print(
        f"  (MODEL_TYPE={model_type}, BODYREGION_TYPE={bodyregion_type}, "
        f"REGION_PREFIX={region_prefix})",
        file=sys.stderr,
    )

    # Validate checkpoint exists
    checkpoint_path = os.path.join(checkpoint_dir, model_name, f"{epoch}_net_G.pth")
    if not os.path.isfile(checkpoint_path):
        print(f"WARNING: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        print("         Inference step will fail if not skipped.", file=sys.stderr)

    # Output model name to stdout (for shell script to capture)
    print(model_name)


def main():
    parser = argparse.ArgumentParser(
        description="Determine model name based on model type, body region, and region prefix."
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["CUT", "cycleGAN", "pix2pix"],
        help="Model type (CUT, cycleGAN, pix2pix)",
    )
    parser.add_argument(
        "--bodyregion-type",
        required=True,
        choices=["allregions", "regionspecific"],
        help="Body region type (allregions or regionspecific)",
    )
    parser.add_argument(
        "--init-dir",
        required=True,
        help="Directory containing region_prefix.txt (1initNifti)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--epoch",
        required=True,
        help="Epoch number for checkpoint validation",
    )

    args = parser.parse_args()

    find_model(
        args.model_type,
        args.bodyregion_type,
        args.init_dir,
        args.checkpoint_dir,
        args.epoch,
    )


if __name__ == "__main__":
    main()
