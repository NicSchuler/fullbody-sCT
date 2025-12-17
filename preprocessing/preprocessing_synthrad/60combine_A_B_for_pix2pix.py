"""
Usage:
    python 60combine_A_B_for_pix2pix.py [normalization_method]

Examples:
    python 60combine_A_B_for_pix2pix.py 31baseline
    python 60combine_A_B_for_pix2pix.py 32p99
    python 60combine_A_B_for_pix2pix.py 33nyul
    python 60combine_A_B_for_pix2pix.py 34npeaks

If no argument is provided, uses default: 32p99
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


# Default normalization method
NORMALIZATION_METHOD = "32p99"  # Default

# Base directory
BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")

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
    
    ROOT = BASE_ROOT / method / "6materialized_splits" / "pix2pix"
    
    if not ROOT.exists():
        raise FileNotFoundError(
            f"Input directory not found: {ROOT}\n"
            f"Please run 50_split_folderstructure.py {method} first"
        )
    
    print("=" * 60)
    print(f"Normalization method: {method}")
    print(f"Input root: {ROOT}")
    print("=" * 60)

def concat_nifti_pair(path_a: Path, path_b: Path, out_path: Path):
    """Load two NIfTI files, concatenate vertically (A on top, B below), save as NIfTI."""
    img_a = nib.load(str(path_a))
    img_b = nib.load(str(path_b))

    data_a = img_a.get_fdata()
    data_b = img_b.get_fdata()

    if data_a.shape != data_b.shape:
        raise ValueError(
            f"Shape mismatch for {path_a.name} and {path_b.name}: "
            f"{data_a.shape} vs {data_b.shape}"
        )
    

    # to concatenate we want the data to be 2d
    # (256x256x1) -> (256x256)
    if data_a.ndim == 3 and data_a.shape[-1] == 1:
        data_a = data_a[..., 0]
        data_b = data_b[..., 0]

    # HORIZONTAL concatenation: axis=1 -> A on top, B on bottom
    # change axis=0 for VERTICAL concatenation
    # # For 256x256 inputs -> 256x512 
    data_ab = np.concatenate([data_a, data_b], axis=1)

    # add 3rd dimension again
    # (256x256) -> (256x256x1)
    data_ab = data_ab[..., np.newaxis]

    header = img_a.header.copy()
    header.set_data_shape(data_ab.shape)

    img_ab = nib.Nifti1Image(data_ab, img_a.affine, header)
    nib.save(img_ab, str(out_path))


def create_pairs(root: Path):
    splits = ["train", "val", "test"]

    for split in splits:
        folder_a = root / split / "A"
        folder_b = root / split / "B"
        out_dir = root / "AB" / split

        if not folder_a.exists():
            print(f"Skipping split '{split}' (no folder {folder_a})")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing split: {split} ===")
        print(f"A folder: {folder_a}")
        print(f"B folder: {folder_b}")
        print(f"Output : {out_dir}")

        # Support .nii and .nii.gz
        files_a = sorted(list(folder_a.glob("*.nii")) + list(folder_a.glob("*.nii.gz")))

        if not files_a:
            print(f"  No NIfTI files found in {folder_a}, skipping.")
            continue

        for path_a in tqdm(files_a):
            path_b = folder_b / path_a.name
            if not path_b.exists():
                print(f"  WARNING: no matching file in B for {path_a.name}, skipping.")
                continue

            out_path = out_dir / path_a.name
            concat_nifti_pair(path_a, path_b, out_path)
            #print(f"  Saved: {out_path.name}")


def main():
    global NORMALIZATION_METHOD
    
    # Parse command line argument if provided
    if len(sys.argv) > 1:
        NORMALIZATION_METHOD = sys.argv[1]
    
    # Configure paths based on normalization method
    configure_paths(NORMALIZATION_METHOD)
    
    create_pairs(ROOT)
    print(f"\nCompleted creating A+B pairs for {NORMALIZATION_METHOD}")


if __name__ == "__main__":
    main()