#!/usr/bin/env python3
"""Step 20: run TotalSegmentator on shared CT folders (manifest-free).

Reads CT from:
  <export-root>/test_patients_shared/<patient>/CT/*.nii*
Writes masks to:
  <export-root>/test_patients_shared/<patient>/TS_CT/
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from totalsegmentator.python_api import totalsegmentator
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "TotalSegmentator python API is required for step 20. "
        "Install/activate env with `totalsegmentator` package."
    ) from exc


DEFAULT_PREPROC_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
DEFAULT_OUTPUT_BASE = DEFAULT_PREPROC_ROOT / "11dvhEvalCases"

TOTAL_TASK_ROIS = ["liver", "kidney_left", "kidney_right", "spinal_cord"]
BODY_TASK_ROIS = ["skin"]
REQUIRED_ROIS = TOTAL_TASK_ROIS + BODY_TASK_ROIS


def has_roi(out_dir: Path, roi_name: str) -> bool:
    return (out_dir / f"{roi_name}.nii.gz").exists() or (out_dir / f"{roi_name}.nii").exists()


def has_all_rois(out_dir: Path, rois: list[str]) -> bool:
    return all(has_roi(out_dir, roi) for roi in rois)


def missing_rois(out_dir: Path, rois: list[str]) -> list[str]:
    return [roi for roi in rois if not has_roi(out_dir, roi)]


def read_patients(patients: list[str] | None, patients_file: Path | None, shared_root: Path) -> list[str]:
    if patients:
        return list(dict.fromkeys(patients))

    if patients_file:
        lines = [ln.strip() for ln in patients_file.read_text().splitlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]

    return sorted([p.name for p in shared_root.iterdir() if p.is_dir()]) if shared_root.is_dir() else []


def find_shared_ct(patient_dir: Path) -> Path:
    ct_dir = patient_dir / "CT"
    if not ct_dir.is_dir():
        raise FileNotFoundError(f"Missing CT folder: {ct_dir}")

    cands = sorted(ct_dir.glob("*.nii.gz")) + sorted(ct_dir.glob("*.nii"))
    if not cands:
        raise FileNotFoundError(f"No CT NIfTI found in {ct_dir}")
    return cands[0]


def run_task(ct_nifti: Path, out_dir: Path, task: str, device: str, fast: bool, roi_subset=None) -> None:
    kwargs = {
        "task": task,
        "device": device,
        "fast": fast,
    }
    if roi_subset:
        kwargs["roi_subset"] = roi_subset
    totalsegmentator(str(ct_nifti), str(out_dir), **kwargs)


def run_ct_task_flow(ct_nifti: Path, out_dir: Path, device: str, fast: bool, force: bool) -> dict:
    task_state = {
        "ran_total_task": False,
        "ran_body_task": False,
        "skipped_total_existing": False,
        "skipped_body_existing": False,
        "missing_before_run": [],
        "missing_after_run": [],
    }

    task_state["missing_before_run"] = missing_rois(out_dir, REQUIRED_ROIS)

    if not force and has_all_rois(out_dir, TOTAL_TASK_ROIS):
        task_state["skipped_total_existing"] = True
    else:
        run_task(
            ct_nifti=ct_nifti,
            out_dir=out_dir,
            task="total",
            device=device,
            fast=fast,
            roi_subset=TOTAL_TASK_ROIS,
        )
        task_state["ran_total_task"] = True

    if not force and has_all_rois(out_dir, BODY_TASK_ROIS):
        task_state["skipped_body_existing"] = True
    else:
        run_task(
            ct_nifti=ct_nifti,
            out_dir=out_dir,
            task="body",
            device=device,
            fast=fast,
        )
        task_state["ran_body_task"] = True

    task_state["missing_after_run"] = missing_rois(out_dir, REQUIRED_ROIS)
    return task_state


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 20 - run TotalSegmentator on shared CT folders")
    p.add_argument("--export-root", type=Path, default=DEFAULT_OUTPUT_BASE)
    p.add_argument("--patients", nargs="*", default=None)
    p.add_argument("--patients-file", type=Path, default=None)
    p.add_argument("--device", default="gpu")
    p.add_argument("--fast", action="store_true", help="Use TotalSegmentator fast mode")
    p.add_argument("--force", action="store_true")
    return p


def main() -> None:
    args = make_parser().parse_args()
    export_root = args.export_root.resolve()
    shared_root = export_root / "test_patients_shared"

    patients = read_patients(args.patients, args.patients_file, shared_root)
    if not patients:
        raise RuntimeError(f"No patients found under {shared_root}. Run step 10 first.")

    ok = 0
    failed = 0

    for patient in patients:
        patient_dir = shared_root / patient
        out_dir = patient_dir / "TS_CT"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            ct_nifti = find_shared_ct(patient_dir)
            task_state = run_ct_task_flow(
                ct_nifti=ct_nifti,
                out_dir=out_dir,
                device=args.device,
                fast=args.fast,
                force=args.force,
            )
            if task_state["missing_after_run"]:
                raise RuntimeError(f"Missing ROI outputs: {task_state['missing_after_run']}")

            ok += 1
            print(
                f"[OK] {patient} | "
                f"total={'run' if task_state['ran_total_task'] else 'skip'} "
                f"body={'run' if task_state['ran_body_task'] else 'skip'}"
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {patient}: {exc}")

    print("\nDone.")
    print(f"Processed: {ok} | Failed: {failed}")


if __name__ == "__main__":
    main()
