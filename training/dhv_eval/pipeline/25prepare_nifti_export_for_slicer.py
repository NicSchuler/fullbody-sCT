#!/usr/bin/env python3
"""Prepare download-friendly NIfTI folders for local Slicer conversion.

Layout under <export-root>:
  test_patients_shared/
    patients/<patient>/
      CT/<patient>_CT.nii.gz
      TS_CT/<totalsegmentator masks from output_dirs.ts_ct>

  <model_name>/
    sCT/<patient>/<patient>_sCT.nii.gz

This keeps real CT and TS masks shared once per patient, while storing model-specific sCT
separately per model.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _pipeline_common import load_manifest


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 25 - prepare shared CT/TS + per-model sCT export folders")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="Single dhv_pipeline_manifest.json")
    group.add_argument(
        "--manifests-root",
        type=Path,
        help="Root to scan recursively for dhv_pipeline_manifest.json files",
    )
    p.add_argument(
        "--export-root",
        type=Path,
        required=True,
        help="Export root (e.g. .../11dvhEvalCases).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing target files/folders.")
    return p


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def infer_model_name(manifest_path: Path) -> str:
    # Expected: <model_name>/reconstruction/dhv_pipeline_manifest.json
    if manifest_path.parent.name == "reconstruction":
        return manifest_path.parent.parent.name
    return manifest_path.parent.name


def prepare_from_manifest(manifest_path: Path, export_root: Path, force: bool) -> tuple[int, int]:
    manifest = load_manifest(manifest_path.resolve())
    cases = manifest.get("cases", [])
    if not cases:
        print(f"[WARN] No cases in {manifest_path}")
        return 0, 0

    model_name = infer_model_name(manifest_path)
    shared_patients_root = export_root / "test_patients_shared" / "patients"
    model_sct_root = export_root / model_name / "sCT"

    print(f"\nManifest: {manifest_path}")
    print(f"Model:    {model_name}")
    print(f"Shared:   {shared_patients_root}")
    print(f"sCT out:  {model_sct_root}")

    ok = 0
    for case in cases:
        patient = case["patient"]
        ct_src = Path(case["ct_nifti"])
        sct_src = Path(case["sct_nifti"])
        ts_src_dir = Path(case["output_dirs"]["ts_ct"])

        # Shared patient folder: CT + TS
        shared_patient_dir = shared_patients_root / patient
        ct_dst = shared_patient_dir / "CT" / f"{patient}_CT.nii.gz"
        ts_dst_dir = shared_patient_dir / "TS_CT"

        if force and ts_dst_dir.exists():
            shutil.rmtree(ts_dst_dir)

        copied_ct = copy_if_exists(ct_src, ct_dst)
        copied_ts = 0
        if ts_src_dir.exists():
            for mask in sorted(ts_src_dir.glob("*.nii*")):
                if copy_if_exists(mask, ts_dst_dir / mask.name):
                    copied_ts += 1

        # Model-specific folder: sCT only
        sct_dst = model_sct_root / patient / f"{patient}_sCT.nii.gz"
        copied_sct = copy_if_exists(sct_src, sct_dst)

        if copied_ct and copied_sct:
            ok += 1
            print(f"[OK]   {patient} | CT:1 sCT:1 TS:{copied_ts}")
        else:
            print(
                f"[WARN] {patient} | CT:{int(copied_ct)} sCT:{int(copied_sct)} TS:{copied_ts} "
                f"(check source paths)"
            )

    return ok, len(cases)


def main() -> None:
    args = make_parser().parse_args()
    export_root = args.export_root.resolve()
    export_root.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        manifest_paths = [args.manifest.resolve()]
    else:
        root = args.manifests_root.resolve()
        manifest_paths = sorted(root.rglob("dhv_pipeline_manifest.json"))
        if not manifest_paths:
            raise RuntimeError(f"No dhv_pipeline_manifest.json found under {root}")

    total_ok = 0
    total_cases = 0
    for manifest_path in manifest_paths:
        ok, count = prepare_from_manifest(manifest_path, export_root, args.force)
        total_ok += ok
        total_cases += count

    print(f"\nDone. Prepared {total_ok}/{total_cases} cases across {len(manifest_paths)} manifest(s).")


if __name__ == "__main__":
    main()
