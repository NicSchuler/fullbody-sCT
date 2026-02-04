#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib


def read_patients(patients: Optional[List[str]], patients_file: Optional[Path], ct_root: Path, sct_root: Path) -> List[str]:
    if patients:
        return list(dict.fromkeys(patients))

    if patients_file:
        lines = [ln.strip() for ln in patients_file.read_text().splitlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]

    ct_ids = {p.name for p in ct_root.iterdir() if p.is_dir()} if ct_root.is_dir() else set()
    sct_ids = {p.name for p in sct_root.iterdir() if p.is_dir()} if sct_root.is_dir() else set()
    return sorted(ct_ids & sct_ids)


def _pick_first(candidates: List[Path]) -> Optional[Path]:
    return candidates[0] if candidates else None


def find_ct_nifti(ct_root: Path, patient: str, ct_subdir: str) -> Path:
    base = ct_root / patient
    checks = [
        base / ct_subdir,
        base,
    ]
    for folder in checks:
        if not folder.is_dir():
            continue
        cands = sorted(folder.glob("*CT*.nii.gz")) + sorted(folder.glob("*CT*.nii"))
        if not cands:
            cands = sorted(folder.glob("*.nii.gz")) + sorted(folder.glob("*.nii"))
        pick = _pick_first(cands)
        if pick:
            return pick

    # fallback recursive
    cands = sorted((ct_root / patient).rglob("*.nii.gz")) + sorted((ct_root / patient).rglob("*.nii"))
    pick = _pick_first(cands)
    if pick:
        return pick
    raise FileNotFoundError(f"CT NIfTI not found for patient {patient} under {ct_root}")


def find_sct_nifti(sct_root: Path, patient: str) -> Path:
    patient_dir = sct_root / patient
    preferred = patient_dir / "sCT_original_dim_reconstructed_alignment.nii.gz"
    if preferred.exists():
        return preferred

    checks = [patient_dir, sct_root]
    for folder in checks:
        if not folder.exists():
            continue
        cands = sorted(folder.glob(f"{patient}*.nii.gz")) + sorted(folder.glob(f"{patient}*.nii"))
        if not cands and folder == patient_dir:
            cands = sorted(folder.glob("*.nii.gz")) + sorted(folder.glob("*.nii"))
        pick = _pick_first(cands)
        if pick:
            return pick

    # fallback recursive under patient folder
    cands = sorted(patient_dir.rglob("*.nii.gz")) + sorted(patient_dir.rglob("*.nii"))
    pick = _pick_first(cands)
    if pick:
        return pick
    raise FileNotFoundError(f"sCT NIfTI not found for patient {patient} under {sct_root}")


def load_grid_info(nifti_path: Path) -> Dict[str, Any]:
    img = nib.load(str(nifti_path))
    return {
        "shape": list(img.shape[:3]),
        "spacing": [float(v) for v in img.header.get_zooms()[:3]],
        "affine": [[float(v) for v in row] for row in img.affine.tolist()],
    }


def save_manifest(manifest: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())
