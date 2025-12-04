from totalsegmentator.python_api import totalsegmentator
import os
import glob
from multiprocessing import get_context
from tqdm import tqdm
NUM_PROCESSES = 5


BASE_DIR = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/1initNifti"
#BASE_DIR = "/local/scratch/datasets/FullbodySCT/flavian_subset/1initNifti"



def find_ct_reg_nii(patient_dir: str) -> str | None:
    """
    Look for a CT_reg folder and then a NIfTI inside.
    Prefer files matching *CT_reg.nii*; fall back to any .nii/.nii.gz.
    """
    ct_reg_dir = os.path.join(patient_dir, "CT_reg")
    if not os.path.isdir(ct_reg_dir):
        return None

    # first try the specific pattern
    candidates = sorted(glob.glob(os.path.join(ct_reg_dir, "*CT_reg.nii*")))
    if not candidates:
        # fallback: any nifti in CT_reg
        candidates = sorted(glob.glob(os.path.join(ct_reg_dir, "*.nii*")))

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



def run_totalseg_for_ct(ct_path: str) -> None:
    """
    Run TotalSegmentator on a single CT:
      - liver (from 'total' task)
      - fat tissues (tissue_types task)
      - body mask (body task)
    All results go to CT_reg/totalsegmentator_output/
    """
    ct_reg_dir = os.path.dirname(ct_path)
    output_dir = os.path.join(ct_reg_dir, "totalsegmentator_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Processing CT: {ct_path} ===")
    print(f"Output dir: {output_dir}")

    # 1) Liver
    if has_liver_mask(output_dir):
        print("✅ Liver mask already exists — skipping liver segmentation.")
    else:
        print(" -> Liver (task='total', roi_subset=['liver'])")
        totalsegmentator(
            ct_path,
            output_dir,
            task="total",
            roi_subset=["liver"],
            fast=True,
            device="gpu",
        )

    # 2) Fat (subcutaneous_fat, torso_fat, skeletal_muscle, ...)
    print(" -> Fat & muscle (task='tissue_types')")
    if has_fat_masks(output_dir):
        print("✅ Fat & muscle masks already exist — skipping fat segmentation.")
    else:
        print(" -> Fat & muscle (task='tissue_types')")
        totalsegmentator(
            ct_path,
            output_dir,
            task="tissue_types",
            fast=False,  # fast not allowed here
            device="gpu",
        )

    # 3) Body mask -currently done only if needed
    if False:
        print(" -> Body mask (task='body')")
        totalsegmentator(
            ct_path,
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

    print(f"Found {len(patient_dirs)} CTs to process.")

    for patient_name in tqdm(patient_dirs):
        patient_dir = os.path.join(BASE_DIR, patient_name)

        ct_path = find_ct_reg_nii(patient_dir)
        if ct_path is None:
            print(f"No CT_reg NIfTI found for patient: {patient_name}")
            continue

        run_totalseg_for_ct(ct_path)



if __name__ == "__main__":
    main()