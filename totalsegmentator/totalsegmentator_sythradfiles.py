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
    print(" -> Liver (task='total', roi_subset=['liver'])")
    totalsegmentator(
        ct_path,
        output_dir,
        task="total",
        roi_subset=["liver"],
        fast=True,
        device="gpu",   # change to "cpu" if needed
    )

    # 2) Fat (subcutaneous_fat, torso_fat, skeletal_muscle, ...)
    print(" -> Fat & muscle (task='tissue_types')")
    totalsegmentator(
        ct_path,
        output_dir,
        task="tissue_types",
        fast=False,
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