"""
Create concatenated A+A pairs for pix2pix inference.

Since we don't have CT (B) during inference, duplicate MR (A) to create
the expected aligned format (A|A instead of A|B).

The model only uses the left half (A) for generation anyway.
The aligned_dataset.py will split the 512x256 image at width/2 to get
a 256x256 input tensor.

Usage:
    python 65combine_A_for_pix2pix_inference.py \\
        --input-dir /path/to/slices/A \\
        --output-dir /path/to/pix2pix_inference/test
"""
import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def concat_aa_pair(path_a: Path, out_path: Path):
    """Concatenate MR with itself (A|A) for pix2pix aligned format.

    This mimics the format created by 60combine_A_B_for_pix2pix.py but
    uses MR for both halves since CT is not available during inference.
    """
    img_a = nib.load(str(path_a))
    data_a = img_a.get_fdata()

    # Squeeze if needed: (256, 256, 1) -> (256, 256)
    if data_a.ndim == 3 and data_a.shape[-1] == 1:
        data_a = data_a[..., 0]

    # Horizontal concatenation: A|A -> (256, 512)
    # This matches 60combine_A_B_for_pix2pix.py which uses axis=1
    data_aa = np.concatenate([data_a, data_a], axis=1)

    # Add 3rd dimension back: (256, 512) -> (256, 512, 1)
    data_aa = data_aa[..., np.newaxis]

    header = img_a.header.copy()
    header.set_data_shape(data_aa.shape)

    img_aa = nib.Nifti1Image(data_aa, img_a.affine, header)
    nib.save(img_aa, str(out_path))


def main():
    parser = argparse.ArgumentParser(
        description="Create A|A pairs for pix2pix inference"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing MR slices (*.nii or *.nii.gz)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for A|A pairs"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return 1

    # Clean and recreate output directory
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all NIfTI files
    files = sorted(list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz")))

    if not files:
        print(f"ERROR: No NIfTI files found in {input_dir}")
        return 1

    print("=" * 60)
    print("Creating A|A pairs for pix2pix inference")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(files)}")
    print("=" * 60)

    for f in tqdm(files, desc="Creating pairs"):
        out_path = output_dir / f.name
        concat_aa_pair(f, out_path)

    print(f"\nCompleted: {len(files)} A|A pairs created")
    return 0


if __name__ == "__main__":
    exit(main())
