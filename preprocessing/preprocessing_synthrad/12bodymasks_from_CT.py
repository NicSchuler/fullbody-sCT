import os
import cv2
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_fill_holes
from tqdm import tqdm

THRESHOLD_CT = -400

# from medical-physics-usz/synthetic_CT_generation/preprocessing/new_helpers.py
def get_mask_biggest_contour(mask_ct):
    for i in range(mask_ct.shape[2]):
        inmask = np.expand_dims(mask_ct[:, :, i].astype(np.uint8), axis=2)
        ret, bin_img = cv2.threshold(inmask, 0.5, 1, cv2.THRESH_BINARY)
        (cnts, _) = cv2.findContours(np.expand_dims(bin_img, axis=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # return None, if no contours detected
        if len(cnts) != 0:
            # based on contour area, get the maximum contour which is a body contour
            segmented = max(cnts, key=cv2.contourArea)
            bin_img[bin_img > 0] = 0
            a = cv2.drawContours(np.expand_dims(bin_img, axis=2), [segmented], 0, (255, 255, 255), -1)
            a[a > 0] = 1
            mask_ct[:, :, i] = a.squeeze()

    return mask_ct

# from medical-physics-usz/synthetic_CT_generation/preprocessing/new_helpers.py
def save_nifti_image(data, affine, file_path):
    """
    Save a NIFTI image to the specified path.

    Args:
        data (np.ndarray): Data to save.
        affine (np.ndarray): Affine matrix of the NIFTI image.
        file_path (str): Path where the NIFTI image will be saved.
    """
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, file_path)

# from medical-physics-usz/synthetic_CT_generation/preprocessing/new_helpers.py
# modified with binary closing and hole filling
def get_body_mask_threshold(nii_array,threshold_ct_body_mask):
    mask_ct = np.zeros(nii_array.shape)
    mask_ct[nii_array > threshold_ct_body_mask] = 1
    mask_ct[nii_array <= threshold_ct_body_mask] = 0
    mask_ct = binary_erosion(mask_ct, iterations=2).astype(np.uint8)
    mask_ct = get_mask_biggest_contour(mask_ct)
    mask_ct = binary_dilation(mask_ct, iterations=5).astype(np.int16)

    # add binary closing (on X-Y plane only, required to close holes in nose area)
    structure = np.ones((31, 31, 1), dtype=bool)
    mask_ct = binary_closing(mask_ct, structure=structure, iterations=1)

    # fill 3D holes (if any remaining)
    mask_ct = binary_fill_holes(mask_ct).astype(np.int16)
    return mask_ct

# from medical-physics-usz/synthetic_CT_generation/preprocessing/new_helpers.py
def process_modality_body_mask_thresholding_only(nii_image, path_nifti, outname, threshold_for_body_mask):
    """
    Process CT modality to create and save body mask (no masked input).

    Args:
        nii_image (nib.Nifti1Image): NIFTI image object.
        path_nifti (str): Directory path to save NIFTI files.
        outname (str): File name for output file.
        threshold_for_body_mask (float): Threshold for CT body mask.

    Returns:
        np.ndarray: Masked input array.
    """
    # Get NIFTI array (data) and affine matrix from the NIFTI image object
    nii_array = nii_image.get_fdata()
    affine = nii_image.affine

    # Generate body mask
    mask_threshold = get_body_mask_threshold(nii_array, threshold_for_body_mask)

    # Save the body mask as a NIFTI file
    path_mask_file = os.path.join(path_nifti, outname)
    save_nifti_image(mask_threshold, affine, path_mask_file)

    return mask_threshold


def main(prefix=None, position=0):
    os.chdir("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/1initNifti")
    all_patients = os.listdir()

    if prefix:
        patients = [pat for pat in all_patients if pat.startswith(prefix)]
    else:
        patients = all_patients

    for patient in tqdm(sorted(patients), position=position):
        input_path = f"{patient}/CT_reg/{patient}_CT_reg.nii.gz"
        output_path = f"{patient}/new_masks"
        output_name = f"{patient}_mask_from_CT_treshold.nii.gz"

        nii_image = nib.load(input_path)
        os.makedirs(output_path, exist_ok=True)
        new_mask = process_modality_body_mask_thresholding_only(nii_image, output_path, output_name, THRESHOLD_CT)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Generate body masks from CT for a subset of patients."
    )

    # optional
    parser.add_argument(
        "-p", "--prefix",
        help="Only process patients whose names start with this prefix."
    )

    parser.add_argument(
    "--position", type=int, default=0,
    help="tqdm bar position for parallel runs."
    )

    args = parser.parse_args()
    main(prefix=args.prefix, position=args.position)