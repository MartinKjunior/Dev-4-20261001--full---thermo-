"""Rigid registration of a 4D image (stack of 3D volumes) with itk-elastix.

Volume 0 is the reference. Every other volume is registered rigidly to it and
resampled onto the reference grid. Works with anything nibabel can load
(NIfTI, PAR/REC - pass the .PAR file for PAR/REC).

Requires: pip install itk-elastix nibabel numpy
"""
from __future__ import annotations

from pathlib import Path

import itk
import nibabel as nib
import numpy as np
from tqdm import tqdm


def to_itk(vol, affine):
    """Convert a numpy volume + nibabel (RAS) affine to an ITK (LPS) image.

    A volume with a single slice is converted to a 2D ITK image, because
    elastix crashes on 3D images that are one voxel thick.
    """
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction = affine[:3, :3] / spacing
    origin = affine[:3, 3]
    flip = np.diag([-1.0, -1.0, 1.0])  # RAS -> LPS
    direction = flip @ direction
    origin = flip @ origin

    n = 2 if vol.shape[2] == 1 else 3
    arr = vol[..., 0] if n == 2 else vol
    img = itk.image_from_array(np.ascontiguousarray(arr.T, dtype=np.float32))  # (x,y,z) -> (z,y,x)
    img.SetSpacing([float(s) for s in spacing[:n]])
    img.SetOrigin([float(o) for o in origin[:n]])
    img.SetDirection(itk.matrix_from_array(np.ascontiguousarray(direction[:n, :n])))
    return img


def from_itk(img, shape):
    """Convert an ITK result back to a numpy array of the given (x, y, z) shape."""
    return itk.array_from_image(img).T.reshape(shape)


def rigid_parameter_object():
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap("rigid")  # rigid only, Mattes mutual information
    pm["AutomaticTransformInitialization"] = ["true"]
    pm["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]
    pm["FinalBSplineInterpolationOrder"] = ["1"]
    po.AddParameterMap(pm)
    return po


def register_rigid(path: str | Path) -> nib.Nifti1Image:
    """Rigidly register all volumes of a 4D image to its first volume.

    Parameters
    ----------
    path : str or Path
        4D image readable by nibabel (NIfTI, or the .PAR file of a PAR/REC pair).

    Returns
    -------
    nibabel.Nifti1Image
        4D image on the reference grid; volume 0 is unchanged.
    """
    nii = nib.load(Path(path))
    if nii.ndim != 4:
        raise ValueError(f"Expected a 4D image, got shape {nii.shape}")
    data = np.asarray(nii.dataobj, dtype=np.float32)
    affine = nii.affine

    fixed = to_itk(data[..., 0], affine)
    out = np.empty_like(data)
    out[..., 0] = data[..., 0]  # reference volume unchanged

    for t in tqdm(range(1, data.shape[3]), desc="Registering volumes"):
        moving = to_itk(data[..., t], affine)
        result, _ = itk.elastix_registration_method(
            fixed, moving, parameter_object=rigid_parameter_object(), log_to_console=False
        )
        out[..., t] = from_itk(result, data.shape[:3])

    return nib.Nifti1Image(out, affine)
