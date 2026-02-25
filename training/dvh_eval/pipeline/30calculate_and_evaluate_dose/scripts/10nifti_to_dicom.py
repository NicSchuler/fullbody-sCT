#!/usr/bin/env python3
"""Convert evaluation NIfTI volumes to DICOM series.

Pipeline assumptions for this project:
- Input root: 11dvhEvalCases
- Ground-truth CT and structures: test_patients_shared/<CASE>
- Predicted sCT models: every folder under input root except test_patients_shared.

Outputs:
- <out>/<MODEL>/<CASE>/CT/*.dcm
- <out>/<MODEL>/<CASE>/sCT/*.dcm
- <out>/<MODEL>/<CASE>/RTSTRUCT/rtstruct.dcm (optional)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Iterable

import SimpleITK as sitk


def _uid() -> str:
    """Generate a deterministic-length DICOM UID-like string."""
    base = "2.25"
    value = uuid.uuid4().int
    return f"{base}.{value}"


def _format_dicom_date_time() -> tuple[str, str]:
    now = dt.datetime.now()
    return now.strftime("%Y%m%d"), now.strftime("%H%M%S")


def _sorted_cases(ct_root: Path, sct_root: Path) -> list[str]:
    ct_cases = {p.name for p in ct_root.iterdir() if p.is_dir()}
    sct_cases = {p.name for p in sct_root.iterdir() if p.is_dir()}
    shared = sorted(ct_cases & sct_cases)
    return shared


def _case_output_complete(case_out: Path) -> bool:
    """Return True when a case already has complete DICOM outputs."""
    ct_dir = case_out / "CT"
    sct_dir = case_out / "sCT"
    rtstruct_file = case_out / "RTSTRUCT" / "rtstruct.dcm"

    if not ct_dir.is_dir() or not sct_dir.is_dir():
        return False
    if not any(ct_dir.glob("*.dcm")) or not any(sct_dir.glob("*.dcm")):
        return False
    return rtstruct_file.exists()


def _discover_model_dirs(data_root: Path) -> list[Path]:
    models = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "test_patients_shared":
            continue
        models.append(p)
    return models


def _find_sct_nifti(model_case_dir: Path, case_id: str) -> Path:
    preferred = model_case_dir / f"{case_id}_sCT.nii.gz"
    if preferred.exists():
        return preferred

    all_nii = sorted(model_case_dir.glob("*.nii.gz"))
    if len(all_nii) == 1:
        return all_nii[0]

    sct_like = [p for p in all_nii if "sct" in p.name.lower()]
    if len(sct_like) == 1:
        return sct_like[0]

    if not all_nii:
        raise FileNotFoundError(f"No NIfTI files found in model case folder: {model_case_dir}")
    raise RuntimeError(
        f"Ambiguous sCT NIfTI in {model_case_dir}. "
        f"Expected {case_id}_sCT.nii.gz or a single .nii.gz file, found: {[p.name for p in all_nii]}"
    )


def _write_dicom_series(
    image: sitk.Image,
    out_dir: Path,
    patient_id: str,
    study_uid: str,
    series_uid: str,
    modality: str,
    series_description: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str, time_str = _format_dicom_date_time()
    direction = image.GetDirection()

    # GDCM DICOM writer does not support float pixel buffers for CT output.
    # Store as signed 16-bit with identity rescale.
    image_i16 = sitk.Cast(sitk.Round(image), sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    for z in range(image_i16.GetDepth()):
        slice_img = image_i16[:, :, z]

        # Minimal but valid-enough CT metadata for downstream import.
        slice_img.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.2")  # CT Image Storage
        slice_img.SetMetaData("0008|0060", modality)
        slice_img.SetMetaData("0008|0020", date_str)
        slice_img.SetMetaData("0008|0030", time_str)
        slice_img.SetMetaData("0008|1030", "11dvhEvalCases")
        slice_img.SetMetaData("0008|103e", series_description)
        slice_img.SetMetaData("0010|0010", patient_id)
        slice_img.SetMetaData("0010|0020", patient_id)
        slice_img.SetMetaData("0018|0050", str(image_i16.GetSpacing()[2]))
        slice_img.SetMetaData("0020|000d", study_uid)
        slice_img.SetMetaData("0020|000e", series_uid)
        slice_img.SetMetaData("0020|0011", "1")
        slice_img.SetMetaData("0020|0013", str(z + 1))
        slice_img.SetMetaData("0020|0032", "\\".join(map(str, image_i16.TransformIndexToPhysicalPoint((0, 0, z)))))
        slice_img.SetMetaData(
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        )
        slice_img.SetMetaData("0028|1052", "0")  # Rescale Intercept
        slice_img.SetMetaData("0028|1053", "1")  # Rescale Slope

        writer.SetFileName(str(out_dir / f"{z + 1:04d}.dcm"))
        writer.Execute(slice_img)


def _try_build_rtstruct(case_root: Path, ct_dicom_dir: Path, out_file: Path, case_id: str) -> None:
    """Create RTSTRUCT from binary mask NIfTI files.

    Requires: `pip install nibabel rt-utils`.
    """
    try:
        import nibabel as nib
        import numpy as np
        from rt_utils import RTStructBuilder
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing optional dependencies for RTSTRUCT creation. "
            "Install with: pip install nibabel rt-utils"
        ) from exc

    ts_dir = case_root / "TS_CT"
    if not ts_dir.exists():
        raise FileNotFoundError(f"Missing structures directory: {ts_dir}")

    masks = sorted(ts_dir.glob("*.nii.gz"))
    if not masks:
        raise FileNotFoundError(f"No mask NIfTI files found in: {ts_dir}")

    rtstruct = RTStructBuilder.create_new(
        dicom_series_path=str(ct_dicom_dir),
    )

    for mask_path in masks:
        name = mask_path.name.replace(".nii.gz", "")
        arr = nib.load(str(mask_path)).get_fdata()
        binary = arr > 0.5

        # rt-utils expects mask shape [rows, cols, slices].
        if binary.ndim != 3:
            raise ValueError(f"Mask is not 3D: {mask_path}")

        # NIfTI convention may differ; this transpose works for this dataset layout.
        mask_rcs = np.transpose(binary, (1, 0, 2)).astype(bool)

        if not mask_rcs.any():
            continue

        rtstruct.add_roi(
            mask=mask_rcs,
            name=name,
            description=f"Auto-converted from {mask_path.name}",
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    rtstruct.save(str(out_file))


def _load_reference(case_root: Path) -> dict:
    ref = case_root / "reference_grid.json"
    if not ref.exists():
        return {}
    with ref.open("r", encoding="utf-8") as f:
        return json.load(f)


def _close(a: float, b: float, tol: float = 1e-3) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _validate_image_against_reference(image: sitk.Image, reference: dict, case_id: str, label: str) -> None:
    if not reference:
        return

    expected_shape = tuple(int(v) for v in reference.get("shape", []))
    expected_spacing = tuple(float(v) for v in reference.get("spacing", []))
    affine = reference.get("affine", [])

    if len(expected_shape) == 3:
        got_shape = tuple(int(v) for v in image.GetSize())
        if got_shape != expected_shape:
            raise ValueError(
                f"{case_id} {label}: shape mismatch vs reference_grid.json "
                f"(got {got_shape}, expected {expected_shape})"
            )

    if len(expected_spacing) == 3:
        got_spacing = tuple(float(v) for v in image.GetSpacing())
        for i in range(3):
            if not _close(got_spacing[i], expected_spacing[i]):
                raise ValueError(
                    f"{case_id} {label}: spacing mismatch vs reference_grid.json "
                    f"(got {got_spacing}, expected {expected_spacing})"
                )

    if len(affine) == 4 and all(len(row) == 4 for row in affine):
        def direction_from_affine(aff: list[list[float]]) -> tuple[float, ...]:
            c0 = (float(aff[0][0]), float(aff[1][0]), float(aff[2][0]))
            c1 = (float(aff[0][1]), float(aff[1][1]), float(aff[2][1]))
            c2 = (float(aff[0][2]), float(aff[1][2]), float(aff[2][2]))

            n0 = math.sqrt(c0[0] ** 2 + c0[1] ** 2 + c0[2] ** 2)
            n1 = math.sqrt(c1[0] ** 2 + c1[1] ** 2 + c1[2] ** 2)
            n2 = math.sqrt(c2[0] ** 2 + c2[1] ** 2 + c2[2] ** 2)
            if n0 == 0 or n1 == 0 or n2 == 0:
                raise ValueError(f"{case_id} {label}: invalid affine in reference_grid.json (zero-length axis)")

            return (
                c0[0] / n0,
                c0[1] / n0,
                c0[2] / n0,
                c1[0] / n1,
                c1[1] / n1,
                c1[2] / n1,
                c2[0] / n2,
                c2[1] / n2,
                c2[2] / n2,
            )

        affine_direct = [[float(v) for v in row] for row in affine]
        # Many NIfTI toolchains store affine in RAS, while ITK/SimpleITK reports LPS.
        # Accept either interpretation to avoid false mismatches.
        affine_ras_to_lps = [[float(v) for v in row] for row in affine]
        for col in range(4):
            affine_ras_to_lps[0][col] *= -1.0
            affine_ras_to_lps[1][col] *= -1.0

        candidates = [
            ("direct", affine_direct),
            ("ras_to_lps", affine_ras_to_lps),
        ]

        got_origin = tuple(float(v) for v in image.GetOrigin())
        got_direction = tuple(float(v) for v in image.GetDirection())

        match_found = False
        expected_debug = []
        for name, aff in candidates:
            exp_origin = (float(aff[0][3]), float(aff[1][3]), float(aff[2][3]))
            exp_direction = direction_from_affine(aff)
            expected_debug.append((name, exp_origin, exp_direction))

            origin_ok = all(_close(got_origin[i], exp_origin[i], tol=1e-2) for i in range(3))
            direction_ok = all(_close(got_direction[i], exp_direction[i], tol=1e-2) for i in range(9))
            if origin_ok and direction_ok:
                match_found = True
                break

        if not match_found:
            raise ValueError(
                f"{case_id} {label}: origin/direction mismatch vs reference_grid.json "
                f"(got origin={got_origin}, direction={got_direction}; "
                f"expected candidates={expected_debug})"
            )


def _validate_ct_sct_consistency(ct_img: sitk.Image, sct_img: sitk.Image, case_id: str) -> None:
    if ct_img.GetSize() != sct_img.GetSize():
        raise ValueError(f"{case_id}: CT/sCT size mismatch ({ct_img.GetSize()} vs {sct_img.GetSize()})")

    ct_spacing = ct_img.GetSpacing()
    sct_spacing = sct_img.GetSpacing()
    for i in range(3):
        if not _close(ct_spacing[i], sct_spacing[i]):
            raise ValueError(f"{case_id}: CT/sCT spacing mismatch ({ct_spacing} vs {sct_spacing})")

    ct_origin = ct_img.GetOrigin()
    sct_origin = sct_img.GetOrigin()
    for i in range(3):
        if not _close(ct_origin[i], sct_origin[i], tol=1e-2):
            raise ValueError(f"{case_id}: CT/sCT origin mismatch ({ct_origin} vs {sct_origin})")

    ct_direction = ct_img.GetDirection()
    sct_direction = sct_img.GetDirection()
    for i in range(9):
        if not _close(ct_direction[i], sct_direction[i], tol=1e-2):
            raise ValueError(f"{case_id}: CT/sCT direction mismatch ({ct_direction} vs {sct_direction})")


def convert_case(
    data_root: Path,
    out_root: Path,
    model_name: str,
    case_id: str,
    sct_nii: Path,
) -> None:
    case_root = data_root / "test_patients_shared" / case_id
    ct_nii = case_root / "CT" / f"{case_id}_CT.nii.gz"

    if not ct_nii.exists():
        raise FileNotFoundError(f"Missing CT NIfTI: {ct_nii}")
    if not sct_nii.exists():
        raise FileNotFoundError(f"Missing sCT NIfTI: {sct_nii}")

    reference = _load_reference(case_root)

    case_out = out_root / model_name / case_id
    ct_out = case_out / "CT"
    sct_out = case_out / "sCT"

    study_uid = _uid()

    ct_img = sitk.ReadImage(str(ct_nii))
    sct_img = sitk.ReadImage(str(sct_nii))

    _validate_image_against_reference(ct_img, reference, case_id=case_id, label="CT")
    _validate_image_against_reference(sct_img, reference, case_id=case_id, label="sCT")
    _validate_ct_sct_consistency(ct_img, sct_img, case_id=case_id)

    _write_dicom_series(
        image=ct_img,
        out_dir=ct_out,
        patient_id=case_id,
        study_uid=study_uid,
        series_uid=_uid(),
        modality="CT",
        series_description="Reference CT",
    )

    _write_dicom_series(
        image=sct_img,
        out_dir=sct_out,
        patient_id=case_id,
        study_uid=study_uid,
        series_uid=_uid(),
        modality="CT",
        series_description="Synthetic CT",
    )

    rt_out = case_out / "RTSTRUCT" / "rtstruct.dcm"
    _try_build_rtstruct(case_root=case_root, ct_dicom_dir=ct_out, out_file=rt_out, case_id=case_id)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert 11dvhEvalCases NIfTI files into DICOM series.")
    p.add_argument("--data-root", type=Path, default=Path("11dvhEvalCases"))
    p.add_argument("--out-root", type=Path, default=Path("outputs/dicom"))
    p.add_argument(
        "--model",
        action="append",
        help="Model folder name under data root (repeatable). If omitted, all model folders are processed.",
    )
    p.add_argument("--case", action="append", help="Case ID (repeatable). If omitted, all shared cases are processed.")
    p.add_argument("--build-rtstruct", action="store_true", help="Create RTSTRUCT from TS_CT masks.")
    p.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recreate DICOM outputs even if a complete case output already exists.",
    )
    return p.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    ct_root = args.data_root / "test_patients_shared"

    if not ct_root.exists():
        raise FileNotFoundError(f"Missing data folder: {ct_root}")

    model_dirs = _discover_model_dirs(args.data_root)
    if not model_dirs:
        raise RuntimeError(f"No model folders found in {args.data_root} (excluding test_patients_shared)")

    if args.model:
        requested = set(args.model)
        model_dirs = [p for p in model_dirs if p.name in requested]
        missing = sorted(requested - {p.name for p in model_dirs})
        if missing:
            raise FileNotFoundError(f"Requested model folder(s) not found: {missing}")
    model_dirs = sorted(model_dirs, key=lambda p: p.name)

    for model_dir in model_dirs:
        model_name = model_dir.name
        cases = _sorted_cases(ct_root, model_dir)
        if args.case:
            requested_cases = set(args.case)
            cases = [c for c in cases if c in requested_cases]
        if not cases:
            print(f"[WARN] Skipping model {model_name}: no matching cases")
            continue

        print(f"[INFO] Processing model: {model_name}")
        for case_id in cases:
            model_case_dir = model_dir / case_id
            sct_nii = _find_sct_nifti(model_case_dir, case_id=case_id)
            case_out = args.out_root / model_name / case_id

            if not args.overwrite_existing and _case_output_complete(case_out):
                print(f"[INFO] Skipping existing case output: {case_id} ({model_name})")
                continue

            print(f"[INFO] Converting case: {case_id} ({model_name})")
            convert_case(
                data_root=args.data_root,
                out_root=args.out_root,
                model_name=model_name,
                case_id=case_id,
                sct_nii=sct_nii,
            )
            print(f"[INFO] Done: {case_id} ({model_name})")

    print(f"[INFO] DICOM output root: {args.out_root} (organized as <model>/<case>/...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
