#!/usr/bin/env python3
"""Export matRad dose cubes to DICOM RTDOSE.

Reads per-case `matrad_workspace.mat` files and writes:
- dose_ct.dcm  (from resRef/physicalDose)
- dose_sct.dcm (from resSynth/physicalDose)

Expected layout:
- results root: outputs/dvh_results/<MODEL>/<CASE>/matrad_workspace.mat
- dicom root:   outputs/dicom/<MODEL>/<CASE>/CT/*.dcm
"""

from __future__ import annotations

import argparse
import datetime as dt
import uuid
from pathlib import Path

import h5py
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def _uid() -> str:
    return f"2.25.{uuid.uuid4().int}"


def _discover_models(results_root: Path) -> list[Path]:
    return sorted([p for p in results_root.iterdir() if p.is_dir()])


def _discover_cases(model_results_dir: Path) -> list[Path]:
    return sorted([p for p in model_results_dir.iterdir() if p.is_dir()])


def _load_dose_cube(mat_path: Path, group_name: str) -> np.ndarray:
    with h5py.File(mat_path, "r") as f:
        arr = f[f"{group_name}/physicalDose"][()]

    # matRad saved arrays appear as (z, x, y). DICOM expects (z, rows, cols).
    dose = np.transpose(arr, (0, 2, 1)).astype(np.float32, copy=False)
    return dose


def _load_ct_series(ct_dir: Path) -> list[Dataset]:
    files = sorted(ct_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No CT DICOM files found in {ct_dir}")

    datasets = [pydicom.dcmread(str(p), stop_before_pixels=True) for p in files]

    # Sort by z coordinate for robust frame ordering.
    datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    return datasets


def _build_rtdose(
    dose_zyx: np.ndarray,
    ct_slices: list[Dataset],
    dose_comment: str,
) -> FileDataset:
    first = ct_slices[0]
    rows = int(first.Rows)
    cols = int(first.Columns)
    nframes = len(ct_slices)

    if dose_zyx.shape != (nframes, rows, cols):
        raise ValueError(
            f"Dose cube shape {dose_zyx.shape} incompatible with CT geometry "
            f"(frames={nframes}, rows={rows}, cols={cols})"
        )

    z0 = float(ct_slices[0].ImagePositionPatient[2])
    offsets = np.array([float(ds.ImagePositionPatient[2]) - z0 for ds in ct_slices], dtype=np.float64)

    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"  # RT Dose Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = _uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    now = dt.datetime.now()
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S")

    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTDOSE"

    # Patient/study/series linkage to CT.
    ds.PatientName = getattr(first, "PatientName", "Anonymous")
    ds.PatientID = getattr(first, "PatientID", "Unknown")
    ds.StudyInstanceUID = getattr(first, "StudyInstanceUID", generate_uid())
    ds.StudyDate = getattr(first, "StudyDate", now.strftime("%Y%m%d"))
    ds.StudyTime = getattr(first, "StudyTime", now.strftime("%H%M%S"))
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 300
    ds.InstanceNumber = 1

    ds.StudyDescription = getattr(first, "StudyDescription", "")
    ds.SeriesDescription = f"matRad {dose_comment}"

    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    ds.DoseComment = dose_comment

    ds.ImagePositionPatient = [float(v) for v in first.ImagePositionPatient]
    ds.ImageOrientationPatient = [float(v) for v in first.ImageOrientationPatient]
    ds.PixelSpacing = [float(v) for v in first.PixelSpacing]
    ds.SliceThickness = float(getattr(first, "SliceThickness", offsets[1] - offsets[0] if nframes > 1 else 1.0))

    ds.Rows = rows
    ds.Columns = cols
    ds.NumberOfFrames = nframes
    ds.FrameIncrementPointer = [0x3004000C]  # GridFrameOffsetVector
    ds.GridFrameOffsetVector = offsets.tolist()

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    max_dose = float(np.max(dose_zyx))
    if max_dose <= 0:
        scaling = 1.0
    else:
        scaling = max_dose / 65535.0

    pixel = np.round(dose_zyx / scaling).clip(0, 65535).astype(np.uint16)
    ds.DoseGridScaling = scaling
    ds.PixelData = pixel.tobytes(order="C")

    return ds


def export_case(case_results_dir: Path, case_ct_dir: Path) -> None:
    mat_file = case_results_dir / "matrad_workspace.mat"
    if not mat_file.exists():
        raise FileNotFoundError(f"Missing {mat_file}")

    ct_slices = _load_ct_series(case_ct_dir)

    dose_ct = _load_dose_cube(mat_file, "resRef")
    dose_sct = _load_dose_cube(mat_file, "resSynth")

    ds_ct = _build_rtdose(dose_ct, ct_slices, "CT dose")
    ds_sct = _build_rtdose(dose_sct, ct_slices, "sCT dose")

    out_ct = case_results_dir / "dose_ct.dcm"
    out_sct = case_results_dir / "dose_sct.dcm"

    ds_ct.save_as(str(out_ct), write_like_original=False)
    ds_sct.save_as(str(out_sct), write_like_original=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export RTDOSE DICOM from matRad workspace files.")
    p.add_argument("--results-root", type=Path, default=Path("outputs/dvh_results"))
    p.add_argument("--dicom-root", type=Path, default=Path("outputs/dicom"))
    p.add_argument("--model", action="append", help="Model folder name (repeatable).")
    p.add_argument("--case", action="append", help="Case ID (repeatable).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    model_dirs = _discover_models(args.results_root)
    if args.model:
        requested = set(args.model)
        model_dirs = [m for m in model_dirs if m.name in requested]

    if not model_dirs:
        raise RuntimeError("No result model folders found")

    for model_dir in model_dirs:
        model = model_dir.name
        for case_dir in _discover_cases(model_dir):
            case = case_dir.name
            if args.case and case not in set(args.case):
                continue

            ct_dir = args.dicom_root / model / case / "CT"
            print(f"[INFO] Exporting RTDOSE for {model}/{case}")
            export_case(case_dir, ct_dir)
            print(f"[INFO] Wrote: {case_dir / 'dose_ct.dcm'}")
            print(f"[INFO] Wrote: {case_dir / 'dose_sct.dcm'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
