"""Read the DICOM headers of every training series and save one row per series.

Owner: Taha

Runs in a Kaggle Notebook with the competition data attached (the full data is 311 GB,
so we don't run it locally). Paste this file into one cell and run it.

Output: /kaggle/working/dicom_meta.csv

Only headers are read (stop_before_pixels=True), so no image is loaded.
Set LIMIT to a small number for a test run, and to None for all series.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

LIMIT = 50  # None = all series
OUT_PATH = "/kaggle/working/dicom_meta.csv"

# Kaggle has used both layouts for competition data
ROOT_CANDIDATES = [
    "/kaggle/input/competitions/rsna-intracranial-aneurysm-detection",
    "/kaggle/input/rsna-intracranial-aneurysm-detection",
]

# Tags copied as they are from the first file of each series
SIMPLE_TAGS = [
    "PatientID", "Modality", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "PatientAge", "PatientSex", "Rows", "Columns", "BitsStored", "PhotometricInterpretation",
    "RescaleSlope", "RescaleIntercept", "SliceThickness", "SpacingBetweenSlices",
]

# Tags needed from every file to compute the distance between slices
POSITION_TAGS = ["ImagePositionPatient", "ImageOrientationPatient"]


def first_value(value):
    """WindowCenter, PixelSpacing etc. can hold several values: keep the first one."""
    if value is None:
        return None
    if isinstance(value, pydicom.multival.MultiValue):
        return float(value[0]) if len(value) else None
    return value


def slice_spacing(positions, orientation):
    """Median distance (mm) between neighbouring slices, and the total extent of the volume.

    Each slice's position is projected onto the direction perpendicular to the image plane,
    so the result does not depend on how the scan is tilted.
    """
    if orientation is None or len(positions) < 2:
        return None, None
    row_dir, col_dir = np.array(orientation[:3], float), np.array(orientation[3:], float)
    normal = np.cross(row_dir, col_dir)
    z = np.sort(np.array(positions, float) @ normal)
    gaps = np.diff(z)
    return float(np.median(gaps)), float(z[-1] - z[0])


def multiframe_info(ds):
    """In a multi-frame file, pixel spacing, thickness and positions are inside functional groups."""
    info = {}
    shared = ds.get("SharedFunctionalGroupsSequence")
    per_frame = ds.get("PerFrameFunctionalGroupsSequence")
    groups = [g for g in (shared[0] if shared else None, per_frame[0] if per_frame else None) if g]

    for g in groups:
        if "PixelMeasuresSequence" in g:
            pm = g.PixelMeasuresSequence[0]
            info.setdefault("PixelSpacing", pm.get("PixelSpacing"))
            info.setdefault("SliceThickness", pm.get("SliceThickness"))
        if "PlaneOrientationSequence" in g:
            info.setdefault("ImageOrientationPatient", g.PlaneOrientationSequence[0].get("ImageOrientationPatient"))

    if per_frame:
        info["positions"] = [
            f.PlanePositionSequence[0].ImagePositionPatient
            for f in per_frame if "PlanePositionSequence" in f
        ]
    return info


def read_series(series_dir):
    """One row of metadata for one series folder."""
    series_dir = Path(series_dir)
    row = {"SeriesInstanceUID": series_dir.name}
    try:
        files = sorted(series_dir.glob("*.dcm"))
        row["n_files"] = len(files)
        if not files:
            raise FileNotFoundError(f"no .dcm files in {series_dir}")
        first = pydicom.dcmread(files[0], stop_before_pixels=True)

        for tag in SIMPLE_TAGS:
            row[tag] = first_value(first.get(tag))
        row["WindowCenter"] = first_value(first.get("WindowCenter"))
        row["WindowWidth"] = first_value(first.get("WindowWidth"))
        pixel_spacing = first.get("PixelSpacing")
        orientation = first.get("ImageOrientationPatient")

        n_frames = int(first.get("NumberOfFrames", 1) or 1)
        row["is_multiframe"] = n_frames > 1
        if row["is_multiframe"]:
            # Usually the whole series is this one file
            mf = multiframe_info(first)
            pixel_spacing = pixel_spacing or mf.get("PixelSpacing")
            orientation = orientation or mf.get("ImageOrientationPatient")
            if row["SliceThickness"] is None and mf.get("SliceThickness") is not None:
                row["SliceThickness"] = float(mf["SliceThickness"])
            positions = mf.get("positions", [])
            row["n_slices"] = n_frames * len(files)
        else:
            # One slice per file: read the position of every slice
            positions = []
            for f in files:
                ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=POSITION_TAGS)
                if "ImagePositionPatient" in ds:
                    positions.append(ds.ImagePositionPatient)
            row["n_slices"] = len(files)

        if pixel_spacing is not None:
            row["PixelSpacingRow"], row["PixelSpacingCol"] = float(pixel_spacing[0]), float(pixel_spacing[1])
        row["slice_spacing"], row["z_extent_mm"] = slice_spacing(positions, orientation)
    except Exception as e:  # one broken series should not stop the whole run
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main(root=None, limit=LIMIT, out_path=OUT_PATH):
    root = Path(root or next(p for p in ROOT_CANDIDATES if os.path.exists(p)))
    uids = pd.read_csv(root / "train.csv")["SeriesInstanceUID"].tolist()
    if limit:
        uids = uids[:limit]
    dirs = [root / "series" / uid for uid in uids]

    start = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        rows = []
        for i, row in enumerate(pool.map(read_series, dirs, chunksize=4), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(dirs):
                print(f"{i}/{len(dirs)} series, {time.time() - start:.0f} s")

    meta = pd.DataFrame(rows)
    meta.to_csv(out_path, index=False)
    elapsed = time.time() - start
    print(f"Saved {len(meta)} rows to {out_path} in {elapsed:.0f} s")
    print(f"Estimated time for all 4,348 series: {elapsed / len(meta) * 4348 / 60:.0f} min")
    if "error" in meta:
        print(f"{meta['error'].notna().sum()} series had errors")
    return meta


if __name__ == "__main__":
    main()
