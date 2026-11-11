"""DICOM loading, preprocessing and cached volumes.

Owner: Can

Every series becomes one uint8 volume of shape VOLUME_SHAPE with VOXEL_MM voxels:
1. Read all slices (one file per slice, or multi-frame files) and sort them by position
2. Turn the volume so that its axes are always (top -> bottom, front -> back, right -> left)
3. Cut a fixed box of 160 x 192 x 192 mm that starts at the top of the head
   (long CTAs also contain the neck and chest, which we don't need)
4. Resample the box to 1 mm voxels
5. Scale intensities to 0-255: a fixed HU window for CT, percentiles of the head for MR

The full data is 311 GB, so the cache is made on Kaggle (see kaggle/preprocess_notebook.py):
    python -m src.data --root /kaggle/input/competitions/rsna-intracranial-aneurysm-detection --out /kaggle/working/processed

Output:
    <out>/volumes/<SeriesInstanceUID>.npz   uint8 volume, read it with load_volume()
    <out>/volumes_meta.csv                  one row per series: source spacing, crop box, errors
    <out>/localizers_vox.csv                localizer points in volume coordinates (z, y, x)

Kaggle saves only about 500 files per notebook, so on Kaggle the volumes are written to /kaggle/temp
and pack_volumes() puts them into a few zip files in /kaggle/working. volume_paths() finds the volumes
again, packed or not.

make_mips() turns a cached volume into the three 2D views used by baseline B2.
header_features() reads the model inputs of B1-DICOM (our Kaggle submission) from the DICOM header.
"""
import argparse
import ast
import itertools
import os
import re
import shutil
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage

from src.utils import ID_COL

VOXEL_MM = 1.0
VOLUME_SHAPE = (160, 192, 192)  # (z, y, x) voxels = 160 x 192 x 192 mm
CT_WINDOW = (-100, 700)  # HU mapped to 0 and 255
MR_PERCENTILES = (0.5, 99.9)  # of the head voxels, mapped to 0 and 255
MIN_HEAD_AREA_MM2 = 1000  # slices with less tissue than this are above the head

# MIPs for B2: on CT the skull is brighter than the vessels, so bone is removed before the projection
BONE_LEVEL = 220  # uint8 value on CT, about 590 HU with CT_WINDOW (contrast in vessels is 200-500 HU)
BONE_MARGIN = 3  # voxels removed around bone, its edges are blurred to vessel brightness

# Directions of the volume axes in DICOM patient coordinates (x: right -> left, y: front -> back, z: feet -> head)
CANONICAL_DIRS = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)


def _number(value, default):
    """DICOM numbers can be missing or empty: use the default then."""
    if value is None or value == "":
        return default
    return float(value)


def _frame_attr(groups, ds, sequence, attr):
    """Look up a multi-frame attribute in the functional groups, then at the top level of the file."""
    for group in groups:
        items = group.get(sequence)
        if items and attr in items[0]:
            return items[0].get(attr)
    return ds.get(attr)


def _read_file(path):
    """One dict per 2D image in a DICOM file (several for a multi-frame file)."""
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array
    n_frames = int(ds.get("NumberOfFrames", 1) or 1)
    common = {
        "sop": ds.SOPInstanceUID,
        "modality": ds.get("Modality"),
        "thickness": _number(ds.get("SliceThickness"), None),
        "monochrome1": ds.get("PhotometricInterpretation") == "MONOCHROME1",
    }
    if n_frames == 1:
        return [{
            **common, "frame": 0, "pixels": pixels,
            "position": ds.get("ImagePositionPatient"),
            "orientation": ds.get("ImageOrientationPatient"),
            "pixel_spacing": ds.get("PixelSpacing"),
            "slope": _number(ds.get("RescaleSlope"), 1.0),
            "intercept": _number(ds.get("RescaleIntercept"), 0.0),
        }]

    shared = ds.get("SharedFunctionalGroupsSequence")
    per_frame = ds.get("PerFrameFunctionalGroupsSequence")
    images = []
    for i in range(n_frames):
        groups = [g for g in (per_frame[i] if per_frame else None, shared[0] if shared else None) if g]
        images.append({
            **common, "frame": i, "pixels": pixels[i],
            "position": _frame_attr(groups, ds, "PlanePositionSequence", "ImagePositionPatient"),
            "orientation": _frame_attr(groups, ds, "PlaneOrientationSequence", "ImageOrientationPatient"),
            "pixel_spacing": _frame_attr(groups, ds, "PixelMeasuresSequence", "PixelSpacing"),
            "slope": _number(_frame_attr(groups, ds, "PixelValueTransformationSequence", "RescaleSlope"), 1.0),
            "intercept": _number(_frame_attr(groups, ds, "PixelValueTransformationSequence", "RescaleIntercept"), 0.0),
        })
    return images


class Series:
    """A DICOM series as a float32 volume with canonical axes (top -> bottom, front -> back, right -> left).

    volume:   values after RescaleSlope / RescaleIntercept (HU for CT)
    spacing:  voxel size in mm along (z, y, x)
    modality: 'CT' or 'MR'
    """

    def __init__(self, series_dir):
        images = [img for f in sorted(Path(series_dir).glob("*.dcm")) for img in _read_file(f)]
        if not images:
            raise FileNotFoundError(f"no .dcm files in {series_dir}")
        self.modality = images[0]["modality"]
        self.is_multiframe = any(img["frame"] > 0 for img in images)

        # Keep the most common image size (drops stray images such as scouts)
        size = Counter(img["pixels"].shape for img in images).most_common(1)[0][0]
        self.n_dropped = sum(img["pixels"].shape != size for img in images)
        images = [img for img in images if img["pixels"].shape == size]

        ref = images[0]
        orientation = ref["orientation"] or [1, 0, 0, 0, 1, 0]
        row_dir, col_dir = np.array(orientation[:3], float), np.array(orientation[3:], float)
        normal = np.cross(row_dir, col_dir)

        # Sort slices along the direction perpendicular to the image plane
        if all(img["position"] is not None for img in images):
            pos = np.array([np.array(img["position"], float) @ normal for img in images])
        else:
            pos = np.arange(len(images)) * (ref["thickness"] or 1.0)
        order = np.argsort(pos, kind="stable")
        # Several slices at the same position: keep the first one
        keep = np.r_[True, np.diff(pos[order]) > 1e-3]
        self.n_dropped += int((~keep).sum())
        order = order[keep]
        self._slice_pos = pos[order]
        self._pos_of = {(img["sop"], img["frame"]): p for img, p in zip(images, pos)}

        # In this dataset the pixels of many CTs are already in HU (air is -1024), but some headers
        # still say RescaleIntercept = -1024. Applying it twice would make the brain -1000 HU.
        # Stored values can only be negative after rescaling, so rescale only if none are negative.
        self.raw_min = float(min(images[i]["pixels"].min() for i in order))
        self.rescaled = self.raw_min >= 0
        volume = np.empty((len(order), *size), np.float32)
        for k, i in enumerate(order):
            img = images[i]
            volume[k] = img["pixels"]
            if self.rescaled:
                volume[k] = volume[k] * img["slope"] + img["intercept"]
            if img["monochrome1"]:  # bright means low in MONOCHROME1
                volume[k] *= -1
        del images

        dz = float(np.median(np.diff(self._slice_pos))) if len(order) > 1 else (ref["thickness"] or 1.0)
        pixel_spacing = ref["pixel_spacing"] or [1.0, 1.0]
        spacing = np.array([dz, float(pixel_spacing[0]), float(pixel_spacing[1])])

        # Volume axis 0 follows the slice normal, axis 1 the image rows, axis 2 the image columns.
        # Snap each axis to the closest canonical direction (no oblique resampling).
        dirs = np.stack([normal, col_dir, row_dir])
        self._perm = max(
            itertools.permutations(range(3)),
            key=lambda p: sum(abs(dirs[p[t]] @ CANONICAL_DIRS[t]) for t in range(3)),
        )
        self._flip = [dirs[self._perm[t]] @ CANONICAL_DIRS[t] < 0 for t in range(3)]
        self.plane = ["axial", "coronal", "sagittal"][int(np.argmax(np.abs(normal[[2, 1, 0]])))]

        volume = volume.transpose(self._perm)
        for t in range(3):
            if self._flip[t]:
                volume = np.flip(volume, axis=t)
        self.volume = np.ascontiguousarray(volume)
        self.spacing = spacing[list(self._perm)]

    def locate(self, sop, frame, x, y):
        """Index (z, y, x) in self.volume of pixel (x, y) on an original image, or None if that image was dropped."""
        if (sop, frame) not in self._pos_of:
            return None
        k = int(np.argmin(np.abs(self._slice_pos - self._pos_of[(sop, frame)])))
        src = np.array([k, y, x], float)
        idx = src[list(self._perm)]
        for t in range(3):
            if self._flip[t]:
                idx[t] = self.volume.shape[t] - 1 - idx[t]
        return idx


def _head_box_start(volume, spacing, modality):
    """Start (mm, from voxel 0) of the crop box: top of the head in z, centre of the head in y and x."""
    # Look at the volume at about 2 mm resolution, that is enough to find the head
    step = np.maximum(1, np.round(2.0 / spacing)).astype(int)
    small = volume[::step[0], ::step[1], ::step[2]]
    small_spacing = spacing * step
    threshold = -300 if modality == "CT" else 0.1 * np.percentile(small, 99)
    labels, n = ndimage.label(small > threshold)
    if n == 0:
        return np.zeros(3)
    # The largest connected piece is the patient (drops the table and noise)
    mask = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1

    box_mm = np.array(VOLUME_SHAPE) * VOXEL_MM
    area = mask.sum(axis=(1, 2)) * small_spacing[1] * small_spacing[2]
    head_slices = np.nonzero(area >= MIN_HEAD_AREA_MM2)[0]
    top = head_slices[0] if len(head_slices) else 0
    in_box = mask[top:top + int(np.ceil(box_mm[0] / small_spacing[0]))]

    start = [top * small_spacing[0]]
    for axis, other in [(1, (0, 2)), (2, (0, 1))]:
        idx = np.nonzero(in_box.any(axis=other))[0]
        centre = (idx[0] + idx[-1]) / 2 * small_spacing[axis] if len(idx) else volume.shape[axis] / 2 * spacing[axis]
        start.append(centre - box_mm[axis] / 2)
    return np.array(start)


def _crop_and_resample(volume, spacing, start_mm, fill):
    """Cut the box from the volume (padding with `fill` outside it) and resample it to VOLUME_SHAPE."""
    out_shape = np.array(VOLUME_SHAPE)
    src_start = np.round(start_mm / spacing).astype(int)
    src_size = np.maximum(1, np.round(out_shape * VOXEL_MM / spacing)).astype(int)

    crop = np.full(src_size, fill, np.float32)
    lo = np.maximum(src_start, 0)
    hi = np.minimum(src_start + src_size, volume.shape)
    if np.all(hi > lo):
        crop[tuple(slice(a - s, b - s) for a, b, s in zip(lo, hi, src_start))] = \
            volume[tuple(slice(a, b) for a, b in zip(lo, hi))]

    # Blur before shrinking so that thin vessels are averaged instead of skipped
    factor = src_size / out_shape
    sigma = np.maximum(0, (factor - 1) / 2)
    if sigma.any():
        crop = ndimage.gaussian_filter(crop, sigma, mode="nearest")
    out = ndimage.zoom(crop, out_shape / src_size, order=1, mode="nearest", grid_mode=True)
    return out, src_start, factor


def _to_uint8(volume, modality):
    if modality == "CT":
        lo, hi = CT_WINDOW
    else:
        head = volume[volume > 0.1 * np.percentile(volume, 99)]
        lo, hi = np.percentile(head, MR_PERCENTILES) if head.size else (volume.min(), volume.max())
    scaled = np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.round(scaled * 255).astype(np.uint8)


def preprocess_series(series_dir, localizers=None):
    """Turn one series folder into a uint8 volume of VOLUME_SHAPE.

    localizers: rows of train_localizers.csv for this series (optional).
    Returns the volume, a dict of metadata, and a list of localizer points in volume coordinates.
    """
    series = Series(series_dir)
    fill = -1024.0 if series.modality == "CT" else float(series.volume.min())
    start_mm = _head_box_start(series.volume, series.spacing, series.modality)
    volume, src_start, factor = _crop_and_resample(series.volume, series.spacing, start_mm, fill)
    volume = _to_uint8(volume, series.modality)

    meta = {
        "modality": series.modality,
        "plane": series.plane,
        "is_multiframe": series.is_multiframe,
        "src_shape": "x".join(map(str, series.volume.shape)),
        "n_dropped_slices": series.n_dropped,
        "raw_min": series.raw_min,
        "rescaled": series.rescaled,
        **{f"src_spacing_{a}": round(float(s), 4) for a, s in zip("zyx", series.spacing)},
        **{f"box_start_mm_{a}": round(float(s), 1) for a, s in zip("zyx", start_mm)},
    }

    points = []
    for _, row in (localizers if localizers is not None else pd.DataFrame()).iterrows():
        c = ast.literal_eval(row["coordinates"])
        idx = series.locate(row["SOPInstanceUID"], int(c.get("f", 0)), c["x"], c["y"])
        if idx is None:
            continue
        # Same mapping as the resampling: voxel centres, grid_mode=True
        z, y, x = (idx - src_start + 0.5) / factor - 0.5
        inside = all(0 <= v <= s - 1 for v, s in zip((z, y, x), VOLUME_SHAPE))
        points.append({"SOPInstanceUID": row["SOPInstanceUID"], "location": row["location"],
                       "z": round(z, 2), "y": round(y, 2), "x": round(x, 2), "inside": inside})
    return volume, meta, points


def load_volume(path):
    """Read a cached volume: uint8 array of VOLUME_SHAPE."""
    with np.load(path) as f:
        return f["volume"]


def pack_volumes(processed_dir, out_dir, n_archives=8):
    """Put volumes/*.npz into n_archives zip files volumes_<i>.zip in out_dir and copy the CSVs next to them.

    No compression: the .npz files are already compressed. Each archive keeps the volumes/ folder inside.
    """
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in (processed_dir / "volumes").glob("*.npz") if not p.name.endswith(".tmp.npz"))
    for i in range(n_archives):
        with zipfile.ZipFile(out_dir / f"volumes_{i}.zip", "w", zipfile.ZIP_STORED) as z:
            for f in files[i::n_archives]:
                z.write(f, f"volumes/{f.name}")
    for name in ["volumes_meta.csv", "localizers_vox.csv"]:
        if (processed_dir / name).exists():
            shutil.copy(processed_dir / name, out_dir / name)
    print(f"Packed {len(files)} volumes into {n_archives} archives in {out_dir}")


def volume_paths(folder, extract_to=None):
    """{SeriesInstanceUID: path of its .npz} for every cached volume under folder (at any depth).

    Works for data/processed, and for the Kaggle dataset made from pack_volumes() whether Kaggle has
    unzipped the archives or not. Zip files are extracted to extract_to (default: folder) when needed.
    """
    folder = Path(folder)
    paths = {p.stem: p for p in folder.rglob("*.npz") if not p.name.endswith(".tmp.npz")}
    archives = sorted(folder.rglob("volumes_*.zip"))
    if not paths and archives:
        extract_to = Path(extract_to or folder)
        for archive in archives:
            print(f"Extracting {archive.name} to {extract_to}")
            with zipfile.ZipFile(archive) as z:
                z.extractall(extract_to)
        paths = {p.stem: p for p in extract_to.rglob("*.npz")}
    return paths


def make_mips(volume, is_ct):
    """Three 2D views of a cached volume, for baseline B2.

    Returns (axial, coronal, sagittal) uint8 arrays of shape (3, 192, 192), (3, 160, 192), (3, 160, 192),
    looking from the top, the front and the side. The 3 channels of each view are:
    0. MIP: maximum along the viewing direction (vessels are the brightest thing on CTA and MRA)
    1. MIP of the middle half only: less overlap, gives a bit of depth
    2. mean along the viewing direction: the anatomy around the vessels
    On CT, bone (>= BONE_LEVEL) and BONE_MARGIN voxels around it are set to 0 first. Otherwise the
    skull fills the whole MIP.
    """
    if is_ct:
        bone = ndimage.binary_dilation(volume >= BONE_LEVEL, iterations=BONE_MARGIN)
        volume = np.where(bone, 0, volume).astype(np.uint8)
    views = []
    for axis in range(3):
        n = volume.shape[axis]
        middle = np.take(volume, np.arange(n // 4, n - n // 4), axis=axis)
        views.append(np.stack([
            volume.max(axis=axis),
            middle.max(axis=axis),
            np.round(volume.mean(axis=axis, dtype=np.float32)).astype(np.uint8),
        ]))
    return tuple(views)


# Model inputs of B1-DICOM, all read from the header of one file per series (src.models.fit_b1_dicom).
# At submission time there is no train.csv: age, sex and modality must come from the DICOM tags, and
# the Modality tag only says CT or MR. The other tags tell the MR types apart: MRA is 3D gradient echo
# with short TR/TE, MRI T2 is 2D with long TR/TE and few slices, MRI T1post usually has a contrast agent.
HEADER_FEATURES = [
    "is_ct", "age", "male", "n_slices", "slice_thickness", "pixel_spacing", "rows", "field_strength",
    "mr_3d", "repetition_time", "echo_time", "flip_angle", "contrast",
    "desc_angio", "desc_t1", "desc_t2", "desc_post",
]

# Keywords in SeriesDescription / ProtocolName (upper case)
_DESCRIPTION_FLAGS = {
    "desc_angio": r"TOF|MRA|ANGIO|COW|CTA",
    "desc_t1": r"T1|BRAVO|MPRAGE|MPR|SPGR",
    "desc_t2": r"T2|FLAIR|PROPELLER",
    "desc_post": r"POST|\+C|C\+|GD|GAD|CONTRAST",
}


def _first_float(value):
    """A DICOM number as float; the first one for multi-valued tags, NaN if missing or not a number."""
    if isinstance(value, pydicom.multival.MultiValue):
        value = value[0] if len(value) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _age_years(value):
    """PatientAge such as '045Y' -> 45.0. Missing, empty or not in years -> NaN."""
    match = re.fullmatch(r"0*(\d{1,3})Y", str(value or "").strip().upper())
    age = float(match.group(1)) if match else np.nan
    return age if 0 < age < 120 else np.nan


def header_features(series_dir):
    """One row of B1-DICOM inputs (HEADER_FEATURES) for a series folder, from its first file only.

    Missing tags become NaN; LightGBM handles them. Returns a dict with SeriesInstanceUID (the folder name).
    """
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    row = {ID_COL: series_dir.name, **{name: np.nan for name in HEADER_FEATURES}}
    if not files:
        return row
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)

    frames = _first_float(ds.get("NumberOfFrames"))
    frames = int(frames) if frames >= 1 else 1  # NaN (no tag) compares False
    sex = str(ds.get("PatientSex") or "").strip().upper()
    contrast = str(ds.get("ContrastBolusAgent") or "").strip().upper()
    acquisition = str(ds.get("MRAcquisitionType") or "").strip().upper()
    description = f"{ds.get('SeriesDescription') or ''} {ds.get('ProtocolName') or ''}".upper()

    row.update({
        "is_ct": float(str(ds.get("Modality", "")).upper() == "CT"),
        "age": _age_years(ds.get("PatientAge")),
        "male": {"M": 1.0, "F": 0.0}.get(sex, np.nan),
        "n_slices": float(frames * len(files)),
        "slice_thickness": _first_float(ds.get("SliceThickness")),
        "pixel_spacing": _first_float(ds.get("PixelSpacing")),
        "rows": _first_float(ds.get("Rows")),
        "field_strength": _first_float(ds.get("MagneticFieldStrength")),
        "mr_3d": {"3D": 1.0, "2D": 0.0}.get(acquisition, np.nan),
        "repetition_time": _first_float(ds.get("RepetitionTime")),
        "echo_time": _first_float(ds.get("EchoTime")),
        "flip_angle": _first_float(ds.get("FlipAngle")),
        "contrast": float(contrast not in ("", "NONE", "NO")),
        **{name: float(bool(re.search(pattern, description))) for name, pattern in _DESCRIPTION_FLAGS.items()},
    })
    return row


def header_table(series_dirs, workers=None):
    """header_features() of many series in parallel, as a DataFrame (one row per series)."""
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        rows = list(pool.map(header_features, series_dirs, chunksize=8))
    return pd.DataFrame(rows)


def _process(args):
    """Worker: preprocess one series and save it. Errors are returned, not raised."""
    series_dir, localizers, out_dir = args
    uid = Path(series_dir).name
    start = time.time()
    try:
        volume, meta, points = preprocess_series(series_dir, localizers)
        # Write to a temporary name first: if the session stops mid-write, no broken .npz is left
        # behind (a resumed run skips every series that already has an .npz)
        path = Path(out_dir) / "volumes" / f"{uid}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, volume=volume)
        os.replace(tmp, path)
    except Exception as e:  # one broken series should not stop the whole run
        meta, points = {"error": f"{type(e).__name__}: {e}"}, []
    meta = {ID_COL: uid, **meta, "seconds": round(time.time() - start, 1)}
    return meta, [{ID_COL: uid, **p} for p in points]


def _append_csv(rows, path):
    """Add rows to a CSV, replacing old rows of the same series (so a run can be resumed)."""
    new = pd.DataFrame(rows)
    if path.exists() and len(new):
        old = pd.read_csv(path)
        new = pd.concat([old[~old[ID_COL].isin(new[ID_COL])], new], ignore_index=True)
    if len(new):
        new.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Preprocess DICOM series into cached uint8 volumes.")
    parser.add_argument("--root", default="data/raw", help="folder with train.csv and series/")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--limit", type=int, help="only the first N series (for a test run)")
    parser.add_argument("--shard", default="0/1", help="i/n: process every n-th series starting at i")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    root, out = Path(args.root), Path(args.out)
    (out / "volumes").mkdir(parents=True, exist_ok=True)
    uids = sorted(p.name for p in (root / "series").iterdir() if p.is_dir())
    shard, n_shards = map(int, args.shard.split("/"))
    uids = uids[shard::n_shards][:args.limit]
    done = {p.stem for p in (out / "volumes").glob("*.npz") if not p.name.endswith(".tmp.npz")}
    todo = [uid for uid in uids if uid not in done]
    print(f"{len(uids)} series in this shard, {len(uids) - len(todo)} already done, {len(todo)} to go")

    loc = pd.read_csv(root / "train_localizers.csv") if (root / "train_localizers.csv").exists() else None
    loc_of = dict(tuple(loc.groupby(ID_COL))) if loc is not None else {}
    jobs = [(root / "series" / uid, loc_of.get(uid), out) for uid in todo]

    start = time.time()
    metas, points = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (meta, pts) in enumerate(pool.map(_process, jobs), 1):
            metas.append(meta)
            points += pts
            if i % 50 == 0 or i == len(jobs):
                elapsed = time.time() - start
                print(f"{i}/{len(jobs)} series, {elapsed / 60:.1f} min, ~{elapsed / i * (len(jobs) - i) / 60:.0f} min left")
            if i % 200 == 0:  # save progress in case the session stops
                _append_csv(metas, out / "volumes_meta.csv")
                _append_csv(points, out / "localizers_vox.csv")

    _append_csv(metas, out / "volumes_meta.csv")
    _append_csv(points, out / "localizers_vox.csv")
    errors = sum("error" in m for m in metas)
    print(f"Done in {(time.time() - start) / 60:.1f} min, {errors} errors")


if __name__ == "__main__":
    main()
