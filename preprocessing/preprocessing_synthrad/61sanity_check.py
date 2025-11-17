import nibabel as nib
import numpy as np

A_nii = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splitsNonNormalized/pix2pix/train/A/AB_1ABA005-36.nii"
B_nii = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splitsNonNormalized/pix2pix/train/B/AB_1ABA005-36.nii"
AB_nii = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splitsNonNormalized/pix2pix/AB/train/AB_1ABA005-36.nii"

A = nib.load(A_nii).get_fdata()
B = nib.load(B_nii).get_fdata()
AB = nib.load(AB_nii).get_fdata()

# squeeze last dim if present
if A.ndim == 3 and A.shape[-1] == 1:
    A = A[..., 0]
    B = B[..., 0]
    AB = AB[..., 0]

print("A shape:", A.shape)
print("B shape:", B.shape)
print("AB shape:", AB.shape)

H, W = A.shape
print("AB shape expected:", (H, 2*W))

# check that left half of AB == A and right half == B
left  = AB[:, :W]
right = AB[:, W:]

print("max |A - left| :", np.abs(A - left).max())
print("max |B - right|:", np.abs(B - right).max())
