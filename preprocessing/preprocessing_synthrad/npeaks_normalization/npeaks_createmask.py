import os
import numpy as np
from scipy.ndimage import morphology, measurements, filters, \
    binary_opening, binary_closing, binary_erosion, binary_dilation, binary_fill_holes
import nibabel as nib


'''
1.) I extracted fat masks (body masks you now have you need to save liver masks from totalsegmenetator): https://github.com/medical-physics-usz/synthetic_CT_generation/blob/dba098a1be30ee9b9293da86b6caea3086f53074/preprocessing/CT_MR_preprocessing.py#L81
'''
#parameters to extract fat tissue mask from CT reg for n4+npeaks normalisation
extract_fat_mask = True
threshold_fat_max = -60
threshold_fat_min = -160

def process_fat_mask_extraction(nii_image, body_mask, modality, path_nifti, threshold_fat_min, threshold_fat_max):
    """
    Process fat mask extraction based on modality for the respective modality (CT or MR).
    Returns results with status and any errors encountered.

    Args:
        nii_image (Nifti1Image): Original NIfTI image object.
        body_mask (np.ndarray): Precomputed body mask to apply to extracted fat mask.
        modality (str): Imaging modality ('CT_reg' or 'MR').
        path_nifti (str): Path to save the extracted fat mask.
        threshold_fat_min (float): Minimum threshold for fat in CT scans.
        threshold_fat_max (float): Maximum threshold for fat in CT scans.

    Returns:
        dict: A dictionary with the status of the extraction process for each fat mask.
    """
    results = {}

    try:
        mask_shape = nii_image.shape
        nii_array = nii_image.get_fdata()  # Extract array from NIfTI image

        if modality == "CT_reg":
            try:
                mask_fat = np.zeros(mask_shape)
                mask_fat[nii_array <= threshold_fat_max] = 1
                mask_fat[nii_array < threshold_fat_min] = 0
                mask_fat = binary_opening(mask_fat, iterations=1).astype(np.int16)

                mask_im = nib.Nifti1Image(mask_fat * body_mask, nii_image.affine)
                path_mask_file = os.path.join(path_nifti, '3D_mask_fat.nii')
                nib.save(mask_im, path_mask_file)

                results['fat_CT'] = {'status': 'success'}
            except Exception as e:
                results['fat_CT'] = {'status': 'error', 'error_message': str(e)}

        elif modality == "MR":
            try:
                mask_fat_mr = np.zeros(mask_shape)
                fat_mr_threshold = np.mean(nii_array) * 1.7
                mask_fat_mr[nii_array >= fat_mr_threshold] = 1
                mask_fat_mr = binary_erosion(mask_fat_mr, iterations=1).astype(np.int16)
                mask_fat_mr = binary_opening(mask_fat_mr, iterations=1).astype(np.int16)

                mask_im = nib.Nifti1Image(mask_fat_mr * body_mask, nii_image.affine)
                path_mask_file = os.path.join(path_nifti, '3D_mask_fat_mr.nii')
                nib.save(mask_im, path_mask_file)

                results['fat_MR'] = {'status': 'success'}
            except Exception as e:
                results['fat_MR'] = {'status': 'error', 'error_message': str(e)}

    except Exception as e:
        results['overall_status'] = {'status': 'error', 'error_message': str(e)}

    return results


# Example usage
if extract_fat_mask:
    fat_mask_results = process_fat_mask_extraction( nii_image=nii_image,body_mask=body_mask, modality=modality,path_nifti=path_nifti,
        threshold_fat_min=threshold_fat_min,
        threshold_fat_max=threshold_fat_max
    )
    print(fat_mask_results)
else:
    print("Fat mask extraction skipped.")
