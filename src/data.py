"""DICOM loading, preprocessing and datasets.

Owner: Can

Planned:
- Read a DICOM series (one file per slice, or one multi-frame file), sort slices, build a 3D volume
- Resize volumes to a fixed shape
- Normalize intensity (windowing for CT, percentile scaling for MR)
- Cache volumes as .npy files
"""
