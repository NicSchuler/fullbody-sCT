#!/usr/bin/env python3
"""Step 20: Run TotalSegmentator on CT NIfTI (manifest-driven).

Requested ROI outputs:
- task="total": liver, kidney_left, kidney_right, spinal_cord
- task="body": skin
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline_common import load_manifest, save_manifest

try:
    from totalsegmentator.python_api import totalsegmentator
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "TotalSegmentator python API is required for step 20. "
        "Install/activate env with `totalsegmentator` package."
    ) from exc


TOTAL_TASK_ROIS = ["liver", "kidney_left", "kidney_right", "spinal_cord"]
BODY_TASK_ROIS = ["skin"]
REQUIRED_ROIS = TOTAL_TASK_ROIS + BODY_TASK_ROIS


def has_roi(out_dir: Path, roi_name: str) -> bool:
    return (out_dir / f"{roi_name}.nii.gz").exists() or (out_dir / f"{roi_name}.nii").exists()


def has_all_rois(out_dir: Path, rois: list[str]) -> bool:
    return all(has_roi(out_dir, roi) for roi in rois)


def missing_rois(out_dir: Path, rois: list[str]) -> list[str]:
    return [roi for roi in rois if not has_roi(out_dir, roi)]


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 20 - run TotalSegmentator on CT for selected ROIs")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--device", default="gpu")
    p.add_argument("--fast", action="store_true", help="Use TotalSegmentator fast mode.")
    p.add_argument("--force", action="store_true")
    # Backward-compatible args used by older shell scripts; ignored in new task flow.
    p.add_argument("--task", default=None, help=argparse.SUPPRESS)
    p.add_argument("--extra-tasks", nargs="*", default=None, help=argparse.SUPPRESS)
    p.add_argument("--roi-subset", nargs="*", default=None, help=argparse.SUPPRESS)
    p.add_argument("--expected-roi", default=None, help=argparse.SUPPRESS)
    return p


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
    """Run TotalSegmentator with correct task/ROI mapping."""
    task_state = {
        "ran_total_task": False,
        "ran_body_task": False,
        "skipped_total_existing": False,
        "skipped_body_existing": False,
        "missing_before_run": [],
        "missing_after_run": [],
    }

    task_state["missing_before_run"] = missing_rois(out_dir, REQUIRED_ROIS)

    # total task for liver/kidneys/spinal cord
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

    # body task for skin
    if not force and has_all_rois(out_dir, BODY_TASK_ROIS):
        task_state["skipped_body_existing"] = True
    else:
        run_task(
            ct_nifti=ct_nifti,
            out_dir=out_dir,
            task="body",
            device=device,
            fast=fast
            )
        task_state["ran_body_task"] = True

    task_state["missing_after_run"] = missing_rois(out_dir, REQUIRED_ROIS)
    return task_state


def main() -> None:
    args = make_parser().parse_args()
    manifest = load_manifest(args.manifest.resolve())

    for case in manifest.get("cases", []):
        patient = case["patient"]
        ct_nifti = Path(case["ct_nifti"])
        out_dir = Path(case["output_dirs"]["ts_ct"])
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            task_state = run_ct_task_flow(
                ct_nifti=ct_nifti,
                out_dir=out_dir,
                device=args.device,
                fast=args.fast,
                force=args.force,
            )
            if task_state["missing_after_run"]:
                raise RuntimeError(f"Missing ROI outputs: {task_state['missing_after_run']}")
            print(
                f"[OK] {patient} | "
                f"total={'run' if task_state['ran_total_task'] else 'skip'} "
                f"body={'run' if task_state['ran_body_task'] else 'skip'} "
                f"roi={','.join(REQUIRED_ROIS)}"
            )

            case["status"]["step20_ts_done"] = True
            case["step20"] = {
                "tasks": {
                    "total": TOTAL_TASK_ROIS,
                    "body": BODY_TASK_ROIS,
                },
                "device": args.device,
                "fast": args.fast,
                "force": args.force,
                "output_dir": str(out_dir),
                "results": task_state,
            }
        except Exception as exc:  # noqa: BLE001
            case["status"]["step20_ts_done"] = False
            case.setdefault("errors", []).append(f"step20: {exc}")
            print(f"[FAIL] {patient}: {exc}")

    manifest["step"] = 20
    save_manifest(manifest, args.manifest.resolve())
    print(f"Updated manifest: {args.manifest}")


if __name__ == "__main__":
    main()
