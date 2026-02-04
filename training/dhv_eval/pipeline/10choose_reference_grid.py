#!/usr/bin/env python3
"""Step 10: Choose CT reference grid and create pipeline manifest.

This step is critical: CT NIfTI defines the geometry/reference for subsequent
DICOM/RTSTRUCT generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline_common import (
    find_ct_nifti,
    find_sct_nifti,
    load_grid_info,
    read_patients,
    save_manifest,
)

DEFAULT_PREPROC_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
DEFAULT_CT_ROOT = DEFAULT_PREPROC_ROOT / "1initNifti"
DEFAULT_SCT_BASE = DEFAULT_PREPROC_ROOT / "9latestTestImages"
DEFAULT_OUTPUT_BASE = DEFAULT_PREPROC_ROOT / "11dvhEvalCases"


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 10 - choose CT reference grid and build manifest")
    p.add_argument(
        "--ct-root",
        type=Path,
        default=DEFAULT_CT_ROOT,
        help="CT root (default: 1initNifti; CT is searched in <ct-root>/<patient>/CT_reg/...)",
    )
    p.add_argument(
        "--sct-root",
        type=Path,
        default=None,
        help="Explicit sCT root (<...>/9latestTestImages/<modelName>/reconstruction)",
    )
    p.add_argument(
        "--sct-base",
        type=Path,
        default=DEFAULT_SCT_BASE,
        help="Base for sCT models (default: .../9latestTestImages)",
    )
    p.add_argument(
        "--model-name",
        default=None,
        help="Model folder under --sct-base. If set, sct-root becomes <sct-base>/<model-name>/reconstruction",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for case folders (default: 11dvhEvalCases/<model>/reconstruction if --model-name is set, else 11dvhEvalCases)",
    )
    p.add_argument("--ct-subdir", default="CT_reg", help="CT subfolder name under patient directory")
    p.add_argument("--patients", nargs="*", default=None)
    p.add_argument("--patients-file", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    return p


def main() -> None:
    args = make_parser().parse_args()
    ct_root = args.ct_root.resolve()

    if args.sct_root is not None:
        sct_root = args.sct_root.resolve()
    elif args.model_name:
        sct_root = (args.sct_base / args.model_name / "test_50" / "reconstruction").resolve()
    else:
        raise RuntimeError("Provide either --sct-root or --model-name (with optional --sct-base).")

    if args.output_root:
        output_root = args.output_root.resolve()
    else:
        if args.model_name:
            output_root = (DEFAULT_OUTPUT_BASE / args.model_name / "reconstruction").resolve()
        else:
            output_root = DEFAULT_OUTPUT_BASE.resolve()

    patients = read_patients(args.patients, args.patients_file, ct_root, sct_root)
    if not patients:
        raise RuntimeError("No patients found")

    cases = []
    failed = []

    for patient in patients:
        try:
            ct_nifti = find_ct_nifti(ct_root, patient, args.ct_subdir)
            sct_nifti = find_sct_nifti(sct_root, patient)
            grid = load_grid_info(ct_nifti)

            case_out = output_root / patient
            cases.append(
                {
                    "patient": patient,
                    "ct_nifti": str(ct_nifti),
                    "sct_nifti": str(sct_nifti),
                    "reference_grid": grid,
                    "output_dirs": {
                        "patient_root": str(case_out),
                        "real_dicom": str(case_out / "real_overwritten"),
                        "fake_dicom": str(case_out / "fake"),
                        "plan": str(case_out / "Plan"),
                        "ts_ct": str(case_out / "totalsegmentator_ct"),
                        "rtstruct": str(case_out / "Plan" / "RTSTRUCT_ts.dcm"),
                    },
                    "status": {
                        "step10_reference_ready": True,
                        "step20_ts_done": False,
                        "step30_ct_dicom_done": False,
                        "step40_rtstruct_done": False,
                        "step50_sct_dicom_done": False,
                    },
                }
            )
            print(f"[OK] {patient}")
        except Exception as exc:  # noqa: BLE001
            failed.append({"patient": patient, "error": str(exc)})
            print(f"[FAIL] {patient}: {exc}")

    manifest = {
        "step": 10,
        "ct_root": str(ct_root),
        "sct_root": str(sct_root),
        "output_root": str(output_root),
        "cases": cases,
        "failed": failed,
    }

    manifest_path = args.manifest or (output_root / "dhv_pipeline_manifest.json")
    save_manifest(manifest, manifest_path)
    print(f"\nManifest written: {manifest_path}")
    print(f"Cases: {len(cases)} | Failed: {len(failed)}")


if __name__ == "__main__":
    main()
