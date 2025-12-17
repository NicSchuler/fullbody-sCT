import os
import shutil
from tqdm import tqdm

# === USER CONFIGURATION ===
source_root = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed"       # e.g. "/data/A"
subdir_name = "31baseline/5slices"                                # e.g. "1initNifti" or "31baseline/5slices"
destination_root = "/local/scratch/datasets/FullbodySCT/flavian_subset"     # e.g. "/data/filtered"
patient_ids = ["1ABA005", "1ABA009", "1ABA042", "1ABA063", "1ABA073", "1ABA087", "1ABA108", "1ABA119", "1ABB035", "1ABB056", "1ABB112"]     # define your patient numbers here
valid_extensions = [".nii", ".nii.gz"]  


def has_valid_extension(filename, extensions):
    return any(filename.endswith(ext) for ext in extensions)


def file_matches(filename, patient_ids, valid_extensions):
    """True if filename has valid extension and contains one of the patient IDs."""
    if not has_valid_extension(filename, valid_extensions):
        return False
    return any(pid in filename for pid in patient_ids)


def copy_matching_files_with_structure(source_root, subdir_name, destination_root,
                                       patient_ids, valid_extensions):
    source_dir = os.path.join(source_root, subdir_name)
    dest_base = os.path.join(destination_root, subdir_name)

    if not os.path.isdir(source_dir):
        raise RuntimeError("Source directory does not exist: {}".format(source_dir))

    matched_count = 0

    # Walk through the source subdir
    for root, _, files in tqdm(os.walk(source_dir), desc="Walking source tree"):
        for f in files:
            if not file_matches(f, patient_ids, valid_extensions):
                continue

            # Full path to source file
            src_path = os.path.join(root, f)

            # Path relative to the subdir root (so we preserve structure)
            rel_path = os.path.relpath(src_path, source_dir)

            # Destination path mirrors the structure under destination_root/subdir_name
            dest_path = os.path.join(dest_base, rel_path)

            dest_dir = os.path.dirname(dest_path)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                matched_count += 1

    if matched_count == 0:
        print("No matching files found.")
    else:
        print("\n✅ Done. Copied {} matching files.".format(matched_count))


if __name__ == "__main__":
    copy_matching_files_with_structure(
        source_root, subdir_name, destination_root, patient_ids, valid_extensions
    )