#!/usr/bin/env python3
"""
End-to-end pipeline orchestrator for the fullbody sCT (SynthRAD) project.

Runs the full sequence of preprocessing, training, inference, postprocessing,
and evaluation steps by invoking the corresponding numbered scripts as
subprocesses. Each step can be individually targeted using --start and --end.

Pipeline step order:
    10  - Convert raw MHA/NIfTI to unified 1initNifti layout
    12  - Generate CT body masks via thresholding
    13  - Run TotalSegmentator on init NIfTIs (CT + MR)
    20  - Resample / crop / pad volumes to target XY size
    21  - Build patient-level train/val/test split manifest
    22  - Resample TotalSegmentator masks to match step 20 output
    30  - Intensity normalisation (baseline / p99 / Nyul / N4+LIC)
    40  - Extract 2-D axial slices from 3-D volumes
    50  - Materialise train/val/test folder structure
    60  - Concatenate A+B slice pairs for pix2pix
    70  - Create per-body-region subsets
    80  - Train the chosen GAN model
    90  - Run inference / test
    100 - Reconstruct 3-D sCT volumes from 2-D predictions
    110 - Compute per-volume quantitative metrics
    120 - Resample reconstructed volumes back to original patient space
    130-150 - DVH export steps

Usage:
    python pipeline/run_pipeline.py \\
        --synthrad-data-root /data/SynthRAD \\
        --subfolder-name experiment1 \\
        --preprocessing-method 32p99 \\
        --method pix2pix \\
        --epochs 50 \\
        --batch-size 1 \\
        [--start 10] [--end 150]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROC_DIR = REPO_ROOT / "preprocessing" / "preprocessing_synthrad"
TRAINING_DIR = REPO_ROOT / "training"
TRAINING_CUT_DIR = REPO_ROOT / "training_cut"
POSTPROC_DIR = REPO_ROOT / "postprocessing"
DVH_DIR = TRAINING_DIR / "dvh_eval" / "pipeline"

KNOWN_NORMALIZATION_METHODS = {
    "31baseline",
    "32p99",
    "33nyul",
    "34normalized_n4_03LIC",
    "34normalized_n4_08LIC",
    "34normalized_n4_centerspecific_03LIC",
    "34normalized_n4_centerspecific_08LIC",
}

STEP_ORDER = [
    10,   # raw mha -> 1initNifti
    12,   # CT body masks
    13,   # TotalSegmentator on init NIfTI
    20,   # resampling
    21,   # split manifest
    22,   # resample TotalSegmentator masks
    30,   # normalization
    40,   # slice creation
    50,   # split folder structure
    60,   # combine A+B for pix2pix
    70,   # create body region subsets
    80,   # train
    90,   # test
    100,  # reconstruct sCT volumes
    110,  # compute per-volume metrics
    120,  # reconstruct resampled reference data to original dims
    130,  # DVH step 10
    140,  # DVH step 20
    150,  # DVH step 25
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SynthRAD preprocessing, training, testing, metrics, and DVH export pipeline."
    )
    parser.add_argument(
        "--synthrad-data-root",
        type=Path,
        required=True,
        help="Folder that contains the two SynthRAD challenge trees.",
    )
    parser.add_argument(
        "--subfolder-name",
        required=True,
        help="Logical output folder name to create under --synthrad-data-root.",
    )
    parser.add_argument(
        "--patient-filter",
        nargs="+",
        default=None,
        help="Optional patient IDs/tokens to copy from the raw SynthRAD data.",
    )
    parser.add_argument(
        "--body-part-filter",
        choices=["AB", "HN", "TH", "pelvis", "brain"],
        default=None,
        help="Optional body part filter. If omitted, use all body regions.",
    )
    parser.add_argument("--start", type=int, default=10, help="First pipeline step to run.")
    parser.add_argument("--end", type=int, default=150, help="Last pipeline step to run.")
    parser.add_argument(
        "--preprocessing-method",
        required=True,
        choices=sorted(KNOWN_NORMALIZATION_METHODS),
        help="Normalization/preprocessing method folder name.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["CUT", "cut", "pix2pix", "cycleGAN", "cyclegan", "cylceGAN"],
        help="Training method to run.",
    )
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, required=True, help="Training batch size.")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional explicit experiment name for checkpoints/results.",
    )
    parser.add_argument(
        "--normalized-input-root",
        type=Path,
        default=None,
        help="Optional external 3normalized root for methods whose normalization is already prepared.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed for step 21.")
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios.",
    )
    parser.add_argument(
        "--totalseg-device",
        default="gpu",
        choices=["gpu", "cpu", "mps"],
        help="Device for preprocessing TotalSegmentator steps.",
    )
    parser.add_argument(
        "--dvh-totalseg-device",
        default="gpu",
        choices=["gpu", "cpu", "mps"],
        help="Device for DVH CT TotalSegmentator step.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for subprocess steps.",
    )
    return parser.parse_args()


def normalize_method_name(name: str) -> str:
    lowered = name.lower()
    if lowered == "cut":
        return "CUT"
    if lowered in {"cyclegan", "cyclegan", "cylcegan"}:
        return "cycleGAN"
    return name


def sanitize_token(value: str) -> str:
    chars = []
    for ch in value:
        chars.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(chars).strip("_")


def method_folder_name(method: str) -> str:
    return f"3_{method}"


def derive_scope(body_part_filter: str | None) -> str:
    if body_part_filter is None:
        return "allregions"
    mapping = {
        "AB": "AB",
        "HN": "HN",
        "TH": "TH",
        "pelvis": "pelvis",
        "brain": "brain",
    }
    return mapping[body_part_filter]


def build_experiment_name(args: argparse.Namespace) -> str:
    if args.experiment_name:
        return args.experiment_name
    model_name = normalize_method_name(args.method)
    scope = derive_scope(args.body_part_filter)
    return sanitize_token(f"{args.subfolder_name}_{model_name}_{scope}_{args.preprocessing_method}")


def should_run(step: int, start: int, end: int) -> bool:
    return start <= step <= end


def ensure_step_bounds(start: int, end: int) -> None:
    valid = set(STEP_ORDER)
    if start not in valid:
        raise ValueError(f"Unsupported start step: {start}. Valid steps: {STEP_ORDER}")
    if end not in valid:
        raise ValueError(f"Unsupported end step: {end}. Valid steps: {STEP_ORDER}")
    if STEP_ORDER.index(start) > STEP_ORDER.index(end):
        raise ValueError(f"start={start} must be <= end={end} in pipeline order.")


def discover_task1_roots(data_root: Path, output_root: Path) -> list[Path]:
    roots: list[Path] = []
    for candidate in sorted(data_root.rglob("Task1")):
        if not candidate.is_dir():
            continue
        if output_root in candidate.parents or candidate == output_root:
            continue
        if any((candidate / site).is_dir() for site in ("AB", "HN", "TH", "brain", "pelvis")):
            roots.append(candidate)
    deduped: list[Path] = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(root)
    return deduped


def canonical_patient_token(patient_id: str) -> str:
    if "_" in patient_id:
        return patient_id.split("_", 1)[1]
    return patient_id


def matches_patient_filter(patient_id: str, patient_filter: set[str] | None) -> bool:
    if not patient_filter:
        return True
    return patient_id in patient_filter or canonical_patient_token(patient_id) in patient_filter


def read_image(path: Path) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f"{path} is not 3D")
    return image


def find_input(path_base: Path) -> Path | None:
    for ext in (".mha", ".nii.gz", ".nii"):
        candidate = path_base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def copy_or_convert_image(src: Path, dst: Path, *, cast_uint8: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".mha":
        image = sitk.ReadImage(str(src))
        if cast_uint8:
            image = sitk.Cast(image, sitk.sitkUInt8)
        sitk.WriteImage(image, str(dst), useCompression=True)
        return

    if cast_uint8:
        image = read_image(src)
        image = sitk.Cast(image, sitk.sitkUInt8)
        sitk.WriteImage(image, str(dst), useCompression=True)
        return

    shutil.copy2(src, dst)


def run_conversion(
    task1_roots: Sequence[Path],
    init_root: Path,
    *,
    patient_filter: set[str] | None,
    body_part_filter: str | None,
) -> list[str]:
    selected_patients: list[str] = []
    allowed_sites = {body_part_filter} if body_part_filter else None

    init_root.mkdir(parents=True, exist_ok=True)

    for raw_root in task1_roots:
        for site_dir in sorted(raw_root.iterdir()):
            if not site_dir.is_dir():
                continue
            site_name = site_dir.name
            if allowed_sites and site_name not in allowed_sites:
                continue

            for case_dir in sorted(site_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                patient_id = f"{site_name}_{case_dir.name}"
                if not matches_patient_filter(patient_id, patient_filter):
                    continue

                ct_in = find_input(case_dir / "ct")
                mr_in = find_input(case_dir / "mr")
                mask_in = find_input(case_dir / "mask")
                if ct_in is None or mr_in is None:
                    print(f"[SKIP] {case_dir}: missing CT or MR")
                    continue

                out_case = init_root / patient_id
                ct_out = out_case / "CT_reg" / f"{patient_id}_CT_reg.nii.gz"
                mr_out = out_case / "MR" / f"{patient_id}_MR.nii.gz"
                copy_or_convert_image(ct_in, ct_out)
                copy_or_convert_image(mr_in, mr_out)
                if mask_in is not None:
                    mask_out = out_case / "masks" / f"{patient_id}_mask.nii.gz"
                    copy_or_convert_image(mask_in, mask_out, cast_uint8=True)
                selected_patients.append(patient_id)

    unique_patients = sorted(dict.fromkeys(selected_patients))
    print(f"[STEP 10] Prepared {len(unique_patients)} patient(s) in {init_root}")
    return unique_patients


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    label: str,
) -> None:
    cmd_display = " ".join(str(part) for part in cmd)
    print(f"\n[{label}] {cmd_display}")
    subprocess.run(list(map(str, cmd)), cwd=str(cwd), env=env, check=True)


def print_step_start(step: int, description: str) -> None:
    print(f"\n===== START STEP {step}: {description} =====")


def print_step_end(step: int, description: str) -> None:
    print(f"===== END STEP {step}: {description} =====")


def build_env(base_root: Path, mask_slice_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["SYNTHRAD_BASE_ROOT"] = str(base_root)
    if mask_slice_dir is not None:
        env["SYNTHRAD_MASK_SLICE_DIR"] = str(mask_slice_dir)
    return env


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def load_test_patients(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    patients: list[str] = []
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split", "").strip().lower() != "test":
                continue
            example_path = row.get("example_path", "").strip()
            if example_path:
                patients.append(Path(example_path).name)
    patients = sorted(dict.fromkeys(patients))
    if not patients:
        raise RuntimeError(f"No test patients found in {manifest_path}")
    return patients


def normalization_root(base_root: Path, method: str) -> Path:
    return base_root / method_folder_name(method) / "3normalized"


def verify_normalization_available(args: argparse.Namespace, base_root: Path) -> Path:
    if args.normalized_input_root is not None:
        return args.normalized_input_root

    norm_root = normalization_root(base_root, args.preprocessing_method)
    if not norm_root.exists():
        raise RuntimeError(
            f"Normalization output missing: {norm_root}. "
            f"For {args.preprocessing_method}, either run from step 30 with a supported method "
            f"or provide --normalized-input-root."
        )
    return norm_root


def run_normalization_step(
    args: argparse.Namespace,
    *,
    base_root: Path,
    resampled_root: Path,
    manifest_path: Path,
    normalized_root: Path,
    selected_patients: list[str] | None,
    env_base: dict[str, str],
) -> None:
    method = args.preprocessing_method

    if method == "31baseline":
        cmd = [
            args.python,
            str(PREPROC_DIR / "31baseline_standardization.py"),
            "--src-root", str(resampled_root),
            "--out-root", str(normalized_root),
        ]
        if selected_patients:
            cmd.extend(["--patient-ids", *selected_patients])
        elif args.body_part_filter is None:
            cmd.append("--all-data")
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 30")
        return

    if method == "32p99":
        cmd = [
            args.python,
            str(PREPROC_DIR / "32perfile_p99_standardization.py"),
            "--src-root", str(resampled_root),
            "--out-root", str(normalized_root),
        ]
        if selected_patients:
            cmd.extend(["--patient-ids", *selected_patients])
        elif args.body_part_filter is None:
            cmd.append("--all-data")
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 30")
        return

    if method == "33nyul":
        nyul_base = base_root / method_folder_name("33nyul")
        nyul_ready = nyul_base / "3_1NiftiNyulReady"
        nyul_flat = nyul_base / "3_2normalized"
        stage_cmd = [
            args.python,
            str(PREPROC_DIR / "33_1prepare_for_nyul.py"),
            "--input-root", str(resampled_root),
            "--output-root", str(nyul_ready),
            "--manifest", str(manifest_path),
        ]
        if args.body_part_filter is None:
            stage_cmd.append("--all-data")
        run_command(stage_cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 30")

        nyul_env = env_base.copy()
        nyul_env.update({
            "BASE_ROOT": str(nyul_base),
            "NYUL_READY": str(nyul_ready),
            "TRAIN_ROOT": str(nyul_ready / "trainingforcalc"),
            "VALTEST_ROOT": str(nyul_ready / "valtest"),
            "OUT_ROOT": str(nyul_flat),
            "MODEL_PATH": str(nyul_base / "nyul_model_params.npy"),
        })
        run_command(
            ["bash", str(PREPROC_DIR / "33_2nyul_run.sh")],
            cwd=PREPROC_DIR,
            env=nyul_env,
            label="STEP 30",
        )

        finalize_cmd = [
            args.python,
            str(PREPROC_DIR / "33_3_create_nyul_case_folders.py"),
            "--baseline-root", str(resampled_root),
            "--nyul-root", str(nyul_flat),
            "--out-root", str(normalized_root),
            "--copy-ct",
            "--ct-root", str(resampled_root),
            "--overwrite",
        ]
        if args.body_part_filter == "AB":
            finalize_cmd.append("--abdomen-only")
        run_command(finalize_cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 30")
        return

    if method in {
        "34normalized_n4_03LIC",
        "34normalized_n4_08LIC",
        "34normalized_n4_centerspecific_03LIC",
        "34normalized_n4_centerspecific_08LIC",
    }:
        cmd = [
            args.python,
            str(PREPROC_DIR / "34_npeaks.py"),
            "--method", method,
            "--base-root", str(base_root),
            "--src-root", str(resampled_root),
            "--manifest", str(manifest_path),
            "--out-root", str(normalized_root),
            "--disable-visualization",
        ]
        if selected_patients:
            cmd.extend(["--patient-ids", *selected_patients])
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 30")
        return

    raise RuntimeError(f"Unsupported preprocessing method: {method}")


def get_dataset_roots(base_root: Path, method: str, body_part_filter: str | None, gan_method: str) -> tuple[Path, Path]:
    method_root = base_root / method_folder_name(method)
    if body_part_filter is None:
        pix_root = method_root / "6materialized_splits" / "pix2pix" / "AB"
        cycle_root = method_root / "6materialized_splits" / "cyclegan"
    else:
        pix_root = method_root / "7materialized_splits_BodyRegion" / body_part_filter / "pix2pix" / "AB"
        cycle_root = method_root / "7materialized_splits_BodyRegion" / body_part_filter / "cyclegan"

    if gan_method == "pix2pix":
        return pix_root, pix_root
    return cycle_root / "train", cycle_root / "test"


def train_command(
    args: argparse.Namespace,
    base_root: Path,
    experiment_name: str,
    train_root: Path,
) -> tuple[list[str], Path]:
    checkpoints_dir = base_root / "8checkpoints"
    common = [
        "--dataroot", str(train_root),
        "--checkpoints_dir", str(checkpoints_dir),
        "--name", experiment_name,
        "--input_nc", "1",
        "--output_nc", "1",
        "--batch_size", str(args.batch_size),
        "--preprocess", "None",
        "--n_epochs", str(args.epochs),
        "--n_epochs_decay", "0",
        "--save_epoch_freq", "1",
    ]

    model_method = normalize_method_name(args.method)
    if model_method == "CUT":
        script = TRAINING_CUT_DIR / "train.py"
        cmd = [
            args.python,
            str(script),
            *common,
            "--model", "cut",
            "--CUT_mode", "CUT",
            "--no_html",
            "--print_freq", "100",
        ]
    elif model_method == "pix2pix":
        script = TRAINING_DIR / "train.py"
        cmd = [
            args.python,
            str(script),
            *common,
            "--model", "pix2pix",
            "--direction", "AtoB",
            "--no_html",
            "--print_freq", "100",
        ]
    else:
        script = TRAINING_DIR / "train.py"
        cmd = [
            args.python,
            str(script),
            *common,
            "--model", "cycle_gan",
            "--netG", "unet_256",
            "--netD", "basic",
            "--print_freq", "100",
        ]
    return cmd, script.parent


def test_command(
    args: argparse.Namespace,
    base_root: Path,
    experiment_name: str,
    test_root: Path,
) -> tuple[list[str], Path]:
    checkpoints_dir = base_root / "8checkpoints"
    results_dir = base_root / "100results"
    mask_dir = base_root / method_folder_name(args.preprocessing_method) / "5slices" / "masks"
    common = [
        "--dataroot", str(test_root),
        "--checkpoints_dir", str(checkpoints_dir),
        "--name", experiment_name,
        "--phase", "test",
        "--input_nc", "1",
        "--output_nc", "1",
        "--batch_size", "1",
        "--preprocess", "None",
        "--epoch", str(args.epochs),
        "--results_dir", str(results_dir),
    ]

    model_method = normalize_method_name(args.method)
    if model_method == "CUT":
        script = TRAINING_CUT_DIR / "test_synth.py"
        cmd = [
            args.python,
            str(script),
            *common,
            "--model", "cut",
            "--CUT_mode", "CUT",
        ]
    elif model_method == "pix2pix":
        script = TRAINING_DIR / "test_synth.py"
        cmd = [
            args.python,
            str(script),
            *common,
            "--model", "pix2pix",
            "--direction", "AtoB",
        ]
    else:
        script = TRAINING_DIR / "test_synth.py"
        cmd = [
            args.python,
            str(script),
            *common,
            "--model", "cycle_gan",
            "--netG", "unet_256",
            "--netD", "basic",
        ]
    env = build_env(base_root, mask_slice_dir=mask_dir)
    return cmd, script.parent, env


def main() -> int:
    args = parse_args()
    ensure_step_bounds(args.start, args.end)

    base_root = args.synthrad_data_root / args.subfolder_name
    base_root.mkdir(parents=True, exist_ok=True)

    experiment_name = build_experiment_name(args)
    model_method = normalize_method_name(args.method)
    patient_filter = set(args.patient_filter or [])
    task1_roots = discover_task1_roots(args.synthrad_data_root, base_root)
    if should_run(10, args.start, args.end) and not task1_roots:
        raise RuntimeError(
            f"No Task1 folders found under {args.synthrad_data_root}. "
            "Expected the two SynthRAD challenge trees below this directory."
        )

    print("=" * 72)
    print("SynthRAD Pipeline")
    print("=" * 72)
    print(f"Start step:           {args.start}")
    print(f"End step:             {args.end}")
    print(f"Data root:            {args.synthrad_data_root}")
    print(f"Output subfolder:     {args.subfolder_name}")
    print(f"Pipeline base root:   {base_root}")
    print(f"Preprocessing method: {args.preprocessing_method}")
    print(f"Training method:      {model_method}")
    print(f"Experiment name:      {experiment_name}")
    print("=" * 72)

    init_root = base_root / "1initNifti"
    resampled_root = base_root / "2resampledNifti"
    manifest_path = base_root / "splits_manifest.csv"
    normalized_root = normalization_root(base_root, args.preprocessing_method)

    selected_patients: list[str] | None = None
    if should_run(10, args.start, args.end):
        print_step_start(10, "raw mha -> 1initNifti")
        selected_patients = run_conversion(
            task1_roots,
            init_root,
            patient_filter=patient_filter or None,
            body_part_filter=args.body_part_filter,
        )
        if not selected_patients:
            raise RuntimeError("Step 10 produced no patients. Check patient/body filters and raw data layout.")
        print_step_end(10, "raw mha -> 1initNifti")
    elif patient_filter:
        selected_patients = sorted(
            patient_dir.name
            for patient_dir in init_root.iterdir()
            if patient_dir.is_dir() and matches_patient_filter(patient_dir.name, patient_filter)
        )

    env_base = build_env(base_root)

    if should_run(12, args.start, args.end):
        print_step_start(12, "body masks from CT")
        cmd = [
            args.python,
            str(PREPROC_DIR / "12bodymasks_from_CT.py"),
            "--input-root", str(init_root),
        ]
        if selected_patients:
            cmd.extend(["--patient-ids", *selected_patients])
        elif args.body_part_filter:
            cmd.extend(["--prefix", f"{args.body_part_filter}_"])
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 12")
        print_step_end(12, "body masks from CT")

    if should_run(13, args.start, args.end):
        print_step_start(13, "TotalSegmentator on init NIfTI")
        prefixes = [f"{args.body_part_filter}_"] if args.body_part_filter else ["AB_", "HN_", "TH_", "PA_", "BA_"]
        cmd = [
            args.python,
            str(PREPROC_DIR / "13run_totalsegmentator.py"),
            "--input-root", str(init_root),
            "--device", args.totalseg_device,
            "--prefix",
            *prefixes,
        ]
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 13")
        print_step_end(13, "TotalSegmentator on init NIfTI")

    if should_run(20, args.start, args.end):
        print_step_start(20, "resampling")
        cmd = [
            args.python,
            str(PREPROC_DIR / "20resampling.py"),
            "--src-root", str(init_root),
            "--out-root", str(resampled_root),
        ]
        if selected_patients:
            cmd.extend(["--patient-ids", *selected_patients])
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 20")
        print_step_end(20, "resampling")

    if should_run(21, args.start, args.end):
        print_step_start(21, "train/val/test split manifest")
        cmd = [
            args.python,
            str(PREPROC_DIR / "21datasplit.py"),
            "--input-root", str(resampled_root),
            "--out-manifest", str(manifest_path),
            "--seed", str(args.seed),
            "--ratios",
            *(str(value) for value in args.split_ratios),
        ]
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 21")
        print_step_end(21, "train/val/test split manifest")

    if should_run(22, args.start, args.end):
        print_step_start(22, "resample TotalSegmentator masks")
        cmd = [
            args.python,
            str(PREPROC_DIR / "22resample_totalsegmentator_masks.py"),
            "--src-root", str(init_root),
            "--out-root", str(resampled_root),
        ]
        if selected_patients:
            cmd.extend(["--patient-ids", *selected_patients])
        elif args.body_part_filter is None:
            cmd.append("--all-data")
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 22")
        print_step_end(22, "resample TotalSegmentator masks")

    if should_run(30, args.start, args.end):
        print_step_start(30, "normalization")
        run_normalization_step(
            args,
            base_root=base_root,
            resampled_root=resampled_root,
            manifest_path=manifest_path,
            normalized_root=normalized_root,
            selected_patients=selected_patients,
            env_base=env_base,
        )
        print_step_end(30, "normalization")
    elif args.normalized_input_root is not None and args.normalized_input_root != normalized_root:
        remove_path(normalized_root)
        normalized_root.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(args.normalized_input_root, normalized_root, target_is_directory=True)

    if should_run(40, args.start, args.end):
        print_step_start(40, "slice creation")
        verify_normalization_available(args, base_root)
        cmd = [
            args.python,
            str(PREPROC_DIR / "40slice_creator.py"),
            args.preprocessing_method,
            "--base-root", str(base_root),
        ]
        if selected_patients:
            cmd.extend(["--patient-ids", *selected_patients])
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 40")
        print_step_end(40, "slice creation")

    if should_run(50, args.start, args.end):
        print_step_start(50, "materialize split folder structure")
        method_root = base_root / method_folder_name(args.preprocessing_method)
        slices_root = method_root / "5slices"
        out_dir = method_root / "6materialized_splits"
        cmd = [
            args.python,
            str(PREPROC_DIR / "50_split_folderstructure.py"),
            args.preprocessing_method,
            "--slices-root", str(slices_root),
            "--manifest", str(manifest_path),
            "--out-dir", str(out_dir),
        ]
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 50")
        print_step_end(50, "materialize split folder structure")

    if should_run(60, args.start, args.end):
        print_step_start(60, "combine A+B for pix2pix")
        cmd = [
            args.python,
            str(PREPROC_DIR / "60combine_A_B_for_pix2pix.py"),
            args.preprocessing_method,
            "--base-root", str(base_root),
        ]
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 60")
        print_step_end(60, "combine A+B for pix2pix")

    if should_run(70, args.start, args.end):
        print_step_start(70, "create body-region subsets")
        cmd = [
            args.python,
            str(PREPROC_DIR / "70create_subsets_per_body_region.py"),
            args.preprocessing_method,
            "--base-root", str(base_root),
        ]
        run_command(cmd, cwd=PREPROC_DIR, env=env_base, label="STEP 70")
        print_step_end(70, "create body-region subsets")

    if should_run(80, args.start, args.end) or should_run(90, args.start, args.end):
        train_root, test_root = get_dataset_roots(base_root, args.preprocessing_method, args.body_part_filter, model_method)
        if should_run(80, args.start, args.end):
            print_step_start(80, "training")
            train_cmd, train_cwd = train_command(args, base_root, experiment_name, train_root)
            run_command(train_cmd, cwd=train_cwd, env=env_base, label="STEP 80")
            print_step_end(80, "training")
        if should_run(90, args.start, args.end):
            print_step_start(90, "test/inference")
            test_cmd, test_cwd, test_env = test_command(args, base_root, experiment_name, test_root)
            run_command(test_cmd, cwd=test_cwd, env=test_env, label="STEP 90")
            print_step_end(90, "test/inference")

    test_patients = load_test_patients(manifest_path) if should_run(100, args.start, args.end) or should_run(120, args.start, args.end) or should_run(130, args.start, args.end) or should_run(140, args.start, args.end) or should_run(150, args.start, args.end) else None

    if should_run(100, args.start, args.end):
        print_step_start(100, "reconstruct sCT volumes")
        model_folder = f"{experiment_name}/test_{args.epochs}"
        cmd = [
            args.python,
            str(POSTPROC_DIR / "81sct_volume_reconstructor.py"),
            model_folder,
            "--patient-ids",
            *(test_patients or []),
        ]
        run_command(cmd, cwd=POSTPROC_DIR, env=env_base, label="STEP 100")
        print_step_end(100, "reconstruct sCT volumes")

    if should_run(110, args.start, args.end):
        print_step_start(110, "compute volume metrics")
        result_dir = base_root / "100results" / experiment_name / f"test_{args.epochs}"
        cmd = [
            args.python,
            str(POSTPROC_DIR / "110compute_volume_metrics.py"),
            "--root", str(result_dir),
        ]
        run_command(cmd, cwd=REPO_ROOT, env=env_base, label="STEP 110")
        print_step_end(110, "compute volume metrics")

    if should_run(120, args.start, args.end):
        print_step_start(120, "restore resampled references to original dims")
        cmd = [
            args.python,
            str(POSTPROC_DIR / "82resampled_to_original.py"),
            "--output_dir", str(base_root / "2resampledNifti_reconstructed_dims"),
            "--patients",
            *(test_patients or []),
        ]
        run_command(cmd, cwd=POSTPROC_DIR, env=env_base, label="STEP 120")
        print_step_end(120, "restore resampled references to original dims")

    if should_run(130, args.start, args.end):
        print_step_start(130, "DVH export step 10")
        cmd = [
            args.python,
            str(DVH_DIR / "10choose_reference_grid.py"),
            "--ct-root", str(base_root / "2resampledNifti_reconstructed_dims"),
            "--export-root", str(base_root / "11dvhEvalCases"),
            "--patients",
            *(test_patients or []),
            "--force",
        ]
        run_command(cmd, cwd=DVH_DIR, env=env_base, label="STEP 130")
        print_step_end(130, "DVH export step 10")

    if should_run(140, args.start, args.end):
        print_step_start(140, "DVH export step 20")
        cmd = [
            args.python,
            str(DVH_DIR / "20run_totalsegmentator_on_ct.py"),
            "--export-root", str(base_root / "11dvhEvalCases"),
            "--device", args.dvh_totalseg_device,
            "--patients",
            *(test_patients or []),
        ]
        run_command(cmd, cwd=DVH_DIR, env=env_base, label="STEP 140")
        print_step_end(140, "DVH export step 20")

    if should_run(150, args.start, args.end):
        print_step_start(150, "DVH export step 25")
        cmd = [
            args.python,
            str(DVH_DIR / "25prepare_nifti_export_for_slicer.py"),
            "--sct-base", str(base_root / "9latestTestImages"),
            "--export-root", str(base_root / "11dvhEvalCases"),
            "--model-name", experiment_name,
            "--epoch", str(args.epochs),
            "--patients",
            *(test_patients or []),
            "--force",
        ]
        run_command(cmd, cwd=DVH_DIR, env=env_base, label="STEP 150")
        print_step_end(150, "DVH export step 25")

    metrics_csv = base_root / "100results" / experiment_name / f"test_{args.epochs}" / "test_metrics_over_volume.csv"
    dvh_root = base_root / "11dvhEvalCases"
    print("\nPipeline finished.")
    print(f"Experiment name: {experiment_name}")
    print(f"Expected volume metrics CSV: {metrics_csv}")
    print(f"Expected DVH export root:    {dvh_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
