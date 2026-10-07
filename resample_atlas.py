"""Resample a 3D label atlas onto the grid of a 3D or 4D reference image.

Requires: pip install nibabel numpy
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def resample_atlas(atlas_path: str | Path, ref_path: str | Path) -> nib.Nifti1Image:
    """Resample a 3D atlas onto the spatial grid of a reference image.

    The atlas is resampled in 3D with nearest-neighbour interpolation. If the
    reference has extra dimensions (echoes, volumes, ...) the result is
    broadcast along them so it has the same shape as the reference.

    Parameters
    ----------
    atlas_path : str or Path
        3D label image readable by nibabel.
    ref_path : str or Path
        3D or 4D reference image readable by nibabel (NIfTI, or PAR/REC).

    Returns
    -------
    nibabel.Nifti1Image
        Atlas on the reference grid, same shape as the reference.
    """
    atlas = nib.load(Path(atlas_path))
    ref = nib.load(Path(ref_path))
    if atlas.ndim != 3:
        raise ValueError(f"Expected a 3D atlas, got shape {atlas.shape}")

    out = resample_from_to(atlas, (ref.shape[:3], ref.affine), order=0, cval=0)
    if ref.ndim == 3:
        return out

    data = np.asarray(out.dataobj).astype(np.int16)
    extra = ref.shape[3:]
    data = np.broadcast_to(data.reshape(data.shape + (1,) * len(extra)), ref.shape).copy()

    # The shape changed, so build a new image with a header consistent with the reference
    new_atlas = nib.Nifti1Image(data, ref.affine)
    new_atlas.set_sform(ref.affine, code=1)
    new_atlas.set_qform(ref.affine, code=1)
    return new_atlas
