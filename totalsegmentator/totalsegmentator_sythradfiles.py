from totalsegmentator.python_api import totalsegmentator
import os
import glob
from multiprocessing import get_context
from tqdm import tqdm
NUM_PROCESSES = 5


BASE_DIR = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/1initNifti"
#BASE_DIR = "/local/scratch/datasets/FullbodySCT/flavian_subset/1initNifti"



def find_modality_nii(patient_dir: str, modality_dir: str, name_hint: str) -> str | None:
    """
    Look for a modality folder and then a NIfTI inside.
    Prefer files matching *{name_hint}.nii*; fall back to any .nii/.nii.gz.
    """
    modality_path = os.path.join(patient_dir, modality_dir)
    if not os.path.isdir(modality_path):
        return None

    # first try the specific pattern
    candidates = sorted(glob.glob(os.path.join(modality_path, f"*{name_hint}.nii*")))
    if not candidates:
        # fallback: any nifti in modality folder
        candidates = sorted(glob.glob(os.path.join(modality_path, "*.nii*")))

    return candidates[0] if candidates else None

def has_liver_mask(output_dir: str) -> bool:
    """Check whether the liver mask already exists."""
    return os.path.exists(os.path.join(output_dir, "liver.nii.gz"))

def has_fat_masks(output_dir: str) -> bool:
    """Check whether the fat-related masks already exist."""
    expected = [
        "subcutaneous_fat.nii.gz",
        "torso_fat.nii.gz",
        "skeletal_muscle.nii.gz",
    ]
    return all(os.path.exists(os.path.join(output_dir, f)) for f in expected)



def run_totalseg_for_image(image_path: str, modality_dir: str) -> None:
    """
    Run TotalSegmentator on a single image:
      - liver (from 'total' task)
      - fat tissues (tissue_types task)
      - body mask (body task)
    All results go to {modality_dir}/totalsegmentator_output/
    """
    modality_path = os.path.dirname(image_path)
    output_dir = os.path.join(modality_path, "totalsegmentator_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Processing {modality_dir}: {image_path} ===")
    print(f"Output dir: {output_dir}")

    # 1) Liver
    if has_liver_mask(output_dir):
        print("Liver mask already exists — skipping liver segmentation.")
    else:
        print(" -> Liver (task='total', roi_subset=['liver'])")
        totalsegmentator(
            image_path,
            output_dir,
            task="total",
            roi_subset=["liver"],
            fast=True,
            device="gpu",
        )

    # 2) Fat (subcutaneous_fat, torso_fat, skeletal_muscle, ...)
    print(" -> Fat & muscle (task='tissue_types')")
    if has_fat_masks(output_dir):
        print("Fat & muscle masks already exist — skipping fat segmentation.")
    else:
        print(" -> Fat & muscle (task='tissue_types')")
        totalsegmentator(
            image_path,
            output_dir,
            task="tissue_types",
            fast=False,  # fast not allowed here
            device="gpu",
        )

    # 3) Body mask -currently done only if needed
    if False:
        print(" -> Body mask (task='body')")
        totalsegmentator(
            image_path,
            output_dir,
            task="body",
            fast=True,
            device="gpu",
        )



def main():
    # loop over all patient folders in BASE_DIR
    patient_dirs = [
        os.path.join(BASE_DIR, p)
        for p in sorted(os.listdir(BASE_DIR))
        if p.startswith("AB_") and os.path.isdir(os.path.join(BASE_DIR, p))
    ]

    print(f"Found {len(patient_dirs)} patient folders to process.")

    for patient_name in tqdm(patient_dirs):
        patient_dir = os.path.join(BASE_DIR, patient_name)

        ct_path = find_modality_nii(patient_dir, "CT_reg", "CT_reg")
        if ct_path is None:
            print(f"No CT_reg NIfTI found for patient: {patient_name}")
        else:
            run_totalseg_for_image(ct_path, "CT_reg")

        mr_path = find_modality_nii(patient_dir, "MR", "MR")
        if mr_path is None:
            print(f"No MR NIfTI found for patient: {patient_name}")
        else:
            run_totalseg_for_image(mr_path, "MR")



if __name__ == "__main__":
    main()
