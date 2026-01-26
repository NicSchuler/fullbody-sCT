"""
    Usage:
        python 70create_subsets_per_body_region.py [normalization_method]
    
    Examples:
        python 70create_subsets_per_body_region.py 31baseline
        python 70create_subsets_per_body_region.py 32p99
        python 70create_subsets_per_body_region.py 33nyul
        python 70create_subsets_per_body_region.py 34npeaks
    
    If no argument is provided, uses default: 32p99
    
    This will create body-region-specific subsets from the materialized splits.
    """

import os
import shutil
import sys
from tqdm import tqdm
from pathlib import Path

# Default normalization method
NORMALIZATION_METHOD = "32p99"  # Default

# Base directory
BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2")

# Will be set based on NORMALIZATION_METHOD
ROOT = None


def configure_paths(method: str):
    """Configure paths based on normalization method."""
    global ROOT
    
    valid_methods = ["31baseline", "32p99", "33nyul", "34npeaks"]
    
    if method not in valid_methods:
        raise ValueError(
            f"Invalid normalization method: '{method}'\n"
            f"Valid options: {valid_methods}"
        )
    
    ROOT = str(BASE_ROOT / method / "6materialized_splits")
    
    if not os.path.exists(ROOT):
        raise FileNotFoundError(
            f"Input directory not found: {ROOT}\n"
            f"Please run 50_split_folderstructure.py {method} first"
        )
    
    print("=" * 60)
    print(f"Normalization method: {method}")
    print(f"Input root: {ROOT}")
    print("=" * 60)


##The ROOT directory must already exists in a way that the models can handle it.
##This script makes a copy of the ROOT directoy
## For pix2pix it only copies the file in AB (do not confuse with abodmen body part) --> there are already the concatenated images A & B for training
## For cycleGAN it copies the whole directory
## old directoy


def get_new_root_name(current_root_name: str) -> str:
    """Return '7materialized_splits_BodyRegion' for standardized naming."""
    return "7materialized_splits_BodyRegion"


def copy_by_body_region(original_root: str, new_root: str, rel_path: str):
    """
    Copy files from original_root/rel_path into new_root/<body_region>/rel_path,
    where body_region is taken from filename prefix before first '_'.
    """
    src_dir = os.path.join(original_root, rel_path)
    if not os.path.isdir(src_dir):
        print(f"[WARN] Skipping missing directory: {src_dir}")
        return

    print(f"\nProcessing: {src_dir}")

    for fname in tqdm(os.listdir(src_dir)):
        src = os.path.join(src_dir, fname)
        if not os.path.isfile(src):
            continue

        if "_" not in fname:
            print(f"  [WARN] No '_' in filename, skipping: {fname}")
            continue

        body_region = fname.split("_", 1)[0]  # e.g. 'HN', 'B', 'TH'
        # new_root/body_region/pix2pix/AB/train/...
        target_dir = os.path.join(new_root, body_region, *rel_path.split(os.sep))
        os.makedirs(target_dir, exist_ok=True)

        dst = os.path.join(target_dir, fname)
        shutil.copy2(src, dst)


def main():
    global NORMALIZATION_METHOD
    
    # Parse command line argument if provided
    if len(sys.argv) > 1:
        NORMALIZATION_METHOD = sys.argv[1]
    
    # Configure paths based on normalization method
    configure_paths(NORMALIZATION_METHOD)
    
    # Assume script is run from inside the original root
    original_root = os.path.abspath(ROOT)
    current_root_name = os.path.basename(original_root)
    parent_dir = os.path.dirname(original_root)

    try:
        new_root_name = get_new_root_name(current_root_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    new_root = os.path.join(parent_dir, new_root_name)

    print(f"Original root: {original_root}")
    print(f"New root will be: {new_root}")

    if os.path.exists(new_root):
        print(f"[WARN] Target directory already exists, deleting recursively: {new_root}")
        shutil.rmtree(new_root)

    os.makedirs(new_root)
    print(f"Created new root directory: {new_root}")

    # Relevant subdirectories (relative to original_root)
    # pix2pix: only AB
    pix2pix_rel_dirs = [
        os.path.join("pix2pix", "AB", "train"),
        os.path.join("pix2pix", "AB", "val"),
        os.path.join("pix2pix", "AB", "test"),
        os.path.join("pix2pix", "test", "A"),
        os.path.join("pix2pix", "test", "B"),
        os.path.join("pix2pix", "train", "A"),
        os.path.join("pix2pix", "train", "B"),
        os.path.join("pix2pix", "val", "A"),
        os.path.join("pix2pix", "val", "B"),
    ]

    # cyclegan: full structure
    cyclegan_rel_dirs = [
        os.path.join("cyclegan", "train", "trainA"),
        os.path.join("cyclegan", "train", "trainB"),
        os.path.join("cyclegan", "val",   "valA"),
        os.path.join("cyclegan", "val",   "valB"),
        os.path.join("cyclegan", "test",  "testA"),
        os.path.join("cyclegan", "test",  "testB"),
    ]

    for rel_path in tqdm(pix2pix_rel_dirs + cyclegan_rel_dirs):
        copy_by_body_region(original_root, new_root, rel_path)

    print("\nDone. Body-region-specific datasets are in:")
    print(f"  {new_root}")


if __name__ == "__main__":
    main()
