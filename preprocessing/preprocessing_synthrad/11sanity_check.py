import pathlib
import SimpleITK as sitk
from pathlib import Path

# Unified combined output from 10convert_mha_to_nifti.py
path = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/1initNifti")

def get_first_nifti(folder: pathlib.Path):
    """Return the first NIfTI file found in ``folder``, preferring .nii.gz over .nii.

    Args:
        folder: Directory to search.

    Returns:
        Path to the first matching file, or None if the folder does not exist
        or contains no NIfTI files.
    """
    if not folder.exists():
        return None
    for ext in ("*.nii.gz", "*.nii"):
        files = list(folder.glob(ext))
        if files:
            return files[0]
    return None


def check_case(case_dir: pathlib.Path):
    """Verify that the CT and MR volumes for a case share the same size, spacing, and direction.

    Reads the first NIfTI file found in ``case_dir/CT_reg`` and
    ``case_dir/MR`` and compares their SimpleITK metadata.  Prints a
    one-line status tag for the case:

    * ``[OK]``   – all three properties match.
    * ``[MISS]`` – CT or MR file is absent.
    * ``[MISM]`` – at least one property differs (details printed below).

    Args:
        case_dir: Top-level patient directory (e.g. ``1initNifti/AB_1ABA001``).
    """
    ct_dir = case_dir / "CT_reg"
    mr_dir = case_dir / "MR"

    ct_path = get_first_nifti(ct_dir)
    mr_path = get_first_nifti(mr_dir)

    if ct_path is None or mr_path is None:
        print(f"[MISS] {case_dir.name}: CT_reg or MR missing")
        return

    ct = sitk.ReadImage(str(ct_path))
    mr = sitk.ReadImage(str(mr_path))

    ok_size = ct.GetSize() == mr.GetSize()
    ok_spacing = all(
        abs(c - m) < 1e-6
        for c, m in zip(ct.GetSpacing(), mr.GetSpacing())
    )
    ok_direction = list(ct.GetDirection()) == list(mr.GetDirection())

    if ok_size and ok_spacing and ok_direction:
        print(f"[OK]    {case_dir.name}")
    else:
        print(f"[MISM]  {case_dir.name}")
        print(f"       size     CT {ct.GetSize()} vs MR {mr.GetSize()}")
        print(f"       spacing  CT {ct.GetSpacing()} vs MR {mr.GetSpacing()}")
        print(f"       dir eq?  {ok_direction}")


def main():
    """Run ``check_case`` on every patient subdirectory under ``path``."""
    for case_dir in sorted(path.iterdir()):
        if case_dir.is_dir():
            check_case(case_dir)


if __name__ == "__main__":
    main()
