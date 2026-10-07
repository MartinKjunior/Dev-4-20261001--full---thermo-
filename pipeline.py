import pandas as pd
import logging
from pathlib import Path
from fnmatch import fnmatch
from spirit_phantom.core.vials import _load_vial_configurations
import nibabel as nib
from nibabel.parrec import PARRECImage
import numpy as np
from par_parser import extract_echo_times, extract_diffusion_info
from dipy.align import motion_correction
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import TensorModel
from thermometry import lsq_fit_thermometry_signal_model, calculate_temperature_from_df

from scipy import stats
from scipy.optimize import curve_fit
from scipy.ndimage import binary_erosion
from tqdm import tqdm

FIELD_STRENGTH = 3 #T

THERMOMETRY_RESULTS_PATH = Path("SPIRIT dev thermo 20261006") / "processed" / "thermometry_results.csv"
STANDARD_DATA_PATHS = [
    Path("ground_truth") / "O-3574 Calibration_GSP_PVP_20220331.xlsx",
    Path("ground_truth") / "O-41770_GoldStandPhant_GSP_T1T2_20230125.xlsx",
]

sorted_paths = {}

def run(folder_path: Path):
    vials_config = get_vial_config()
    path_list = list(folder_path.rglob("*.REC")) + list(folder_path.rglob("*.nii.gz"))
    for path in path_list:
        if fnmatch(path.name, "*GRADIENT_FFE_Thermometry_*_*_rigid.nii.gz") and "echo_shift" not in path.name:
            atlas_path = path.parent / f"{path.stem.replace('_rigid', '_atlas_manual_fix')}.gz"
            par = path.parent.parent / path.name.replace('_rigid.nii.gz', '.PAR')
            sorted_paths['thermo'] = (path, atlas_path, par)
        if fnmatch(path.name, "*GRADIENT_FFE_Thermometry_echo_shift_*_*_rigid.nii.gz"):
            atlas_path = path.parent / f"{path.stem.replace('_rigid', '_atlas_manual_fix')}.gz"
            par = path.parent.parent / path.name.replace('_rigid.nii.gz', '.PAR')
            sorted_paths['thermo_shifted'] = (path, atlas_path, par)

        if fnmatch(path.name, "*sT1W_3D_FFE_*deg_*_*.REC"):
            par = path.with_suffix(".PAR")
            deg = int(path.name.split("FFE_")[1].split("deg_")[0])
            atlas_path = path.parent / "processed" / f"{path.stem}_atlas.nii.gz"
            sorted_paths.setdefault('T1_map', {})[deg] = (path, atlas_path, par)

        if fnmatch(path.name, "*Multi_echo_GRASE_T2map*.REC"):
            par = path.with_suffix(".PAR")
            atlas_path = path.parent / "processed" / f"{path.stem}_atlas.nii.gz"
            rigid_path = path.parent / "processed" / f"{path.stem}_rigid.nii.gz"
            sorted_paths['T2_map'] = (rigid_path, atlas_path, par)

        if fnmatch(path.name, "*DTI_SSh*.REC"):
            par = path.with_suffix(".PAR")
            rigid_path = path.parent / "processed" / f"{path.stem}_rigid.nii.gz"
            atlas_path = path.parent / "processed" / f"{path.stem}_atlas.nii.gz"
            sorted_paths['diffusion'] = (rigid_path, atlas_path, par)

        if fnmatch(path.name, "*DTI_SSh*_rigid.nii.gz"):
            par = path.parent.parent / path.name.replace('_rigid.nii.gz', '.PAR')
            atlas_path = path.parent / "processed" / f"{path.stem}_atlas.nii.gz"
            sorted_paths['diffusion'] = (path, atlas_path, par)

    if 'thermo' in sorted_paths and 'thermo_shifted' in sorted_paths:
        process_thermo(sorted_paths['thermo'], sorted_paths['thermo_shifted'], vials_config)
    if 'T1_map' in sorted_paths:
        process_T1_map(sorted_paths['T1_map'], vials_config)
    if 'T2_map' in sorted_paths:
        process_T2_map(sorted_paths['T2_map'], vials_config)
    if 'diffusion' in sorted_paths:
        process_diffusion(sorted_paths['diffusion'], vials_config)
    if 'diffusion' in sorted_paths:
        process_diffusion(sorted_paths['diffusion'], vials_config)

def get_vial_config():
    return _load_vial_configurations()

def get_mask(atlas_path: Path, vials_config: list, vial_id: str):
    """Get a binary mask of the vials in the atlas."""
    atlas = nib.load(atlas_path)
    atlas_data = atlas.get_fdata(dtype=np.float64)
    mask = np.zeros(atlas_data.shape, dtype=bool)
    for vial in vials_config:
        if vial.vial_id == vial_id:
            idx = vial.segment_index
            mask |= (atlas_data == idx)
    return mask

# ====== T1 Mapping START ======

def parse_repetition_time(par_path: Path) -> float:
    """Extract the repetition time from a Philips PAR file header."""
    with open(par_path, "r", errors="ignore") as f:
        for line in f:
            if "Repetition time [ms]" in line:
                TR = float(line.split(":", 1)[1].strip())
                if TR > 1.0:
                    return TR / 1000.0  # convert to seconds
    raise ValueError(f"Repetition time not found in {par_path}")

def refine_vial_labels(labels: np.ndarray, slice_range=(22, 40), erode_voxels=1) -> np.ndarray:
    """Experimental: erode each vial by `erode_voxels` and keep only slices in slice_range (0-based, inclusive, None = all).

    To remove, delete this function and its single call in load_masked_stack.
    """
    out = np.zeros_like(labels)
    for label in np.unique(labels[labels > 0]):
        eroded = binary_erosion(labels == label, iterations=erode_voxels)
        out[eroded] = label
    if slice_range is None:
        print(f"Refined mask: eroded {erode_voxels} voxel(s), all slices")
        return out
    out[..., :slice_range[0]] = 0
    out[..., slice_range[1] + 1:] = 0
    print(f"Refined mask: eroded {erode_voxels} voxel(s), slices {slice_range[0]}-{slice_range[1]} only")
    return out

def load_masked_stack(paths: dict, vials_config: list, characterisation: str):
    """Load per-FA images and atlases (keyed by FA), sorted from lowest to highest FA.

    Only voxels inside vials with `characterisation` (e.g. "T1") available are kept.
    Returns a dict with the sorted FAs (deg), TR (s), the masked 4D data (x, y, z, FA),
    the 3D mask and the 3D map of atlas segment indices (0 outside the mask).
    """
    fas = np.array(list(paths.keys()), dtype=np.float64)
    order = np.argsort(fas)
    fas = fas[order]
    keys = [list(paths.keys())[i] for i in order]

    print(f"Loading {characterisation} images and atlases for flip angles (deg): {fas.tolist()}")
    # "fp" scaling gives true signal intensities, comparable between scans with different scale slopes
    data = np.stack([
        PARRECImage.from_filename(str(paths[k][2]), scaling="fp").get_fdata(dtype=np.float64)
        for k in keys
    ], axis=-1)
    atlas = np.stack([np.rint(nib.load(paths[k][1]).get_fdata()).astype(int) for k in keys], axis=-1)
    tr = parse_repetition_time(paths[keys[0]][2])  # same sequence for all FAs

    segment_indices = [v.segment_index for v in vials_config
                       if characterisation in v.si_traceable_characterisation_available]
    # keep only voxels with the same label in every FA atlas
    labels = atlas[..., 0]
    labels = np.where((atlas == labels[..., None]).all(axis=-1), labels, 0)
    print(f"Building mask from atlases: {len(segment_indices)} vials with {characterisation} available "
          f"(segment indices {sorted(segment_indices)})")
    mask = np.isin(labels, segment_indices)
    labels = np.where(mask, labels, 0)
    labels = refine_vial_labels(labels)
    mask = labels > 0
    print(f"TR = {tr * 1e3:.2f} ms, mask keeps {int(mask.sum())} of {mask.size} voxels "
          f"in {len(np.unique(labels[mask]))} vials")
    return {
        "fas": fas,
        "tr": tr,
        "data": data * mask[..., None],
        "mask": mask,
        "labels": labels,
    }

def ernst_signal(fa_rad: np.ndarray, m0: float, t1: float, tr: float) -> np.ndarray:
    """SPGR signal: S = M0 sin(a) (1 - E1) / (1 - cos(a) E1), E1 = exp(-TR/T1)."""
    e1 = np.exp(-tr / t1)
    return m0 * np.sin(fa_rad) * (1.0 - e1) / (1.0 - np.cos(fa_rad) * e1)

def linear_t1_estimate(signal: np.ndarray, fa_rad: np.ndarray, tr: float):
    """Linearised Ernst fit (Gupta 1977): S/sin(a) = E1 * S/tan(a) + C. signal is (n_voxels, n_fa)."""
    x = signal / np.tan(fa_rad)
    y = signal / np.sin(fa_rad)
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    var = (xc ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = np.clip(np.where(var > 0, (xc * yc).sum(axis=1) / var, 0.5), 1e-6, 1 - 1e-6)
        t1 = -tr / np.log(e1)
        m0 = (y.mean(axis=1) - e1 * x.mean(axis=1)) / (1.0 - e1)
    return t1, m0

def fit_t1(stack: dict, t1_bounds=(1e-3, 6.0)) -> np.ndarray:
    """Voxelwise non-linear Ernst fit (initialised from the linear fit) inside the mask.

    Returns a 3D T1 map in seconds, NaN outside the mask or where the fit failed.
    """
    fa_rad = np.deg2rad(stack["fas"])
    tr = stack["tr"]
    mask = stack["mask"]
    signal = stack["data"][mask]  # (n_voxels, n_fa)

    print(f"Fitting T1 in {len(signal)} voxels: linear fit for initial values")
    t1_init, m0_init = linear_t1_estimate(signal, fa_rad, tr)
    t1_init = np.clip(np.nan_to_num(t1_init, nan=1.0), t1_bounds[0] * 1.01, t1_bounds[1] * 0.99)
    m0_init = np.where(np.isfinite(m0_init) & (m0_init > 0), m0_init, signal.max(axis=1))
    m0_init = np.maximum(m0_init, 1e-3)

    def model(fa, m0, t1):
        return ernst_signal(fa, m0, t1, tr)

    t1_fit = np.full(len(signal), np.nan)
    for i in tqdm(range(len(signal)), desc="T1 nonlinear fit"):
        try:
            popt, _ = curve_fit(model, fa_rad, signal[i], p0=[m0_init[i], t1_init[i]],
                                bounds=([0.0, t1_bounds[0]], [np.inf, t1_bounds[1]]), maxfev=200)
            t1_fit[i] = popt[1]
        except Exception:
            pass  # leave as NaN

    t1_map = np.full(mask.shape, np.nan)
    t1_map[mask] = t1_fit
    return t1_map

def summarise_map(param_map: np.ndarray, labels: np.ndarray, vials_config: list, param: str,
                  scale: float = 1e3, unit: str = "ms") -> pd.DataFrame:
    """Mean and std of a parameter map in each vial present in the labels map.

    Values are multiplied by `scale` and reported in `unit` (default: s -> ms).
    """
    rows = []
    for vial in vials_config:
        in_vial = labels == vial.segment_index
        if not in_vial.any():
            continue
        values = param_map[in_vial] * scale
        rows.append({
            "vial_id": vial.vial_id,
            "product_code": vial.product_code,
            "segment_index": vial.segment_index,
            "n_voxels": int(np.isfinite(values).sum()),
            f"{param}_mean_{unit}": np.nanmean(values),
            f"{param}_std_{unit}": np.nanstd(values),
        })
    return pd.DataFrame(rows)

def summarise_t1(t1_map: np.ndarray, labels: np.ndarray, vials_config: list) -> pd.DataFrame:
    return summarise_map(t1_map, labels, vials_config, "T1")

def load_standard_data(paths: list, param: str = "T1") -> pd.DataFrame:
    """Load the reference (temperature-dependent) data for `param` ("T1" or "T2"), dropping unit rows and blanks."""
    df = pd.concat([pd.read_excel(p) for p in paths], ignore_index=True)
    # product codes in the spreadsheets use a non-breaking hyphen
    df["Order code name"] = df["Order code name"].str.replace("\u2011", "-", regex=False)
    numeric_cols = ["Temperature", f"{param} reported", f"{param} uncertainty"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["Order code name"] + numeric_cols)

def compare_to_standard(results: pd.DataFrame, thermo_csv_path: Path, standard_paths: list, param: str = "T1",
                        unit: str = "ms") -> pd.DataFrame:
    """Compare measured vial `param` ("T1", "T2" or "D") with the quadratic temperature fit of the reference data."""
    standard = load_standard_data(standard_paths, param)
    temperature = pd.read_csv(thermo_csv_path)["temperature_C"].mean()
    print(f"Mean phantom temperature from thermometry: {temperature:.2f} C")
    rows = []
    for r in results.to_dict("records"):
        ref = standard[standard["Order code name"] == r["product_code"]].sort_values("Temperature")
        if len(ref) < 3:
            print(f"Vial {r['vial_id']} ({r['product_code']}): not enough reference data for a quadratic fit")
            continue
        coeffs = np.polyfit(ref["Temperature"], ref[f"{param} reported"], 2)
        expected = np.polyval(coeffs, temperature)
        expected_unc = np.interp(temperature, ref["Temperature"], ref[f"{param} uncertainty"])
        mean, std = r[f"{param}_mean_{unit}"], r[f"{param}_std_{unit}"]
        # combine reference uncertainty with the spread of our measurement in the vial
        sigma = np.hypot(expected_unc, std)
        z = (mean - expected) / sigma
        p_value = 2 * stats.norm.sf(abs(z))
        rows.append({
            "vial_id": r["vial_id"],
            "product_code": r["product_code"],
            "segment_index": r["segment_index"],
            "n_voxels": r["n_voxels"],
            "temperature_C": temperature,
            f"{param}_measured_{unit}": mean,
            f"{param}_std_{unit}": std,
            f"{param}_expected_{unit}": expected,
            f"expected_uncertainty_{unit}": expected_unc,
            "z": z,
            "p_value": p_value,
        })
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return comparison

def compare_t1_to_standard(t1_results: pd.DataFrame, thermo_csv_path: Path, standard_paths: list) -> pd.DataFrame:
    return compare_to_standard(t1_results, thermo_csv_path, standard_paths, "T1")

def process_T1_map(t1_map_paths: dict, vials_config: list):
    print("Processing T1 mapping data")
    stack = load_masked_stack(t1_map_paths, vials_config, "T1")
    t1_map = fit_t1(stack)
    results = summarise_t1(t1_map, stack["labels"], vials_config)
    atlas_path = t1_map_paths[min(t1_map_paths)][1]
    comparison = compare_t1_to_standard(results, THERMOMETRY_RESULTS_PATH, STANDARD_DATA_PATHS)
    out_path = atlas_path.parent / "T1_results.csv"
    comparison.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Saved T1 results and comparison to {out_path}")
    return comparison

# ==== T1 Mapping END ======

# ====== T2 Mapping START ======

def load_t2_stack(paths: tuple, vials_config: list):
    """Load the multi-echo magnitude images (echo axis last, sorted by TE) masked to vials with T2 available.

    The GRASE PAR also holds an extra non-magnitude volume (image type != 0, TE = 0), which is dropped.
    Returns a dict with TEs (s), the masked 4D data (x, y, z, echo), the 3D mask and 3D label map.
    """
    image_path, atlas_path, par_path = paths
    print(f"Loading T2 multi-echo image {image_path.name}")
    echo_times = extract_echo_times(par_path) * 1e-3  # ms -> s, one per image row
    tes = np.unique(echo_times[echo_times > 0])  # magnitude echoes, ascending
    data = nib.load(image_path).get_fdata(dtype=np.float64)  # rigidly registered, volumes in PAR order
    if data.shape[-1] < len(tes):
        raise ValueError(f"Expected at least {len(tes)} volumes in {image_path}, got {data.shape[-1]}")
    data = data[..., :len(tes)]  # volumes are ordered by echo number, so the first ones are the echoes
    atlas = np.rint(nib.load(atlas_path).get_fdata()).astype(int)
    labels = atlas[..., 0] if atlas.ndim == 4 else atlas
    if atlas.ndim == 4:
        labels = np.where((atlas[..., :len(tes)] == labels[..., None]).all(axis=-1), labels, 0)

    segment_indices = [v.segment_index for v in vials_config
                       if "T2" in v.si_traceable_characterisation_available]
    print(f"Building mask from atlas: {len(segment_indices)} vials with T2 available "
          f"(segment indices {sorted(segment_indices)})")
    labels = np.where(np.isin(labels, segment_indices), labels, 0)
    labels = refine_vial_labels(labels, slice_range=None)  # only 17 slices, so no slice restriction
    mask = labels > 0
    print(f"TEs = {(tes * 1e3).tolist()} ms, mask keeps {int(mask.sum())} of {mask.size} voxels "
          f"in {len(np.unique(labels[mask]))} vials")
    return {"tes": tes, "data": data * mask[..., None], "mask": mask, "labels": labels}

def linear_t2_estimate(signal: np.ndarray, tes: np.ndarray):
    """Log-linear fit ln S = ln S0 - TE/T2. signal is (n_voxels, n_echo). Returns T2 (s) and S0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(np.clip(signal, 1e-6, None))
        xc = tes - tes.mean()
        slope = ((y - y.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / (xc ** 2).sum()
        t2 = np.where(slope < 0, -1.0 / slope, np.nan)
        s0 = np.exp(y.mean(axis=1) - slope * tes.mean())
    return t2, s0

def fit_t2(stack: dict, t2_bounds=(1e-3, 3.0)) -> np.ndarray:
    """Voxelwise non-linear mono-exponential fit S = S0 exp(-TE/T2), initialised from the log-linear fit.

    Returns a 3D T2 map in seconds, NaN outside the mask or where the fit failed.
    """
    tes = stack["tes"]
    mask = stack["mask"]
    signal = stack["data"][mask]  # (n_voxels, n_echo)

    print(f"Fitting T2 in {len(signal)} voxels: log-linear fit for initial values")
    t2_init, s0_init = linear_t2_estimate(signal, tes)
    t2_init = np.clip(np.nan_to_num(t2_init, nan=0.1), t2_bounds[0] * 1.01, t2_bounds[1] * 0.99)
    s0_init = np.where(np.isfinite(s0_init) & (s0_init > 0), s0_init, signal.max(axis=1))

    def model(te, s0, t2):
        return s0 * np.exp(-te / t2)

    t2_fit = np.full(len(signal), np.nan)
    for i in tqdm(range(len(signal)), desc="T2 nonlinear fit"):
        try:
            popt, _ = curve_fit(model, tes, signal[i], p0=[s0_init[i], t2_init[i]],
                                bounds=([0.0, t2_bounds[0]], [np.inf, t2_bounds[1]]), maxfev=200)
            t2_fit[i] = popt[1]
        except Exception:
            pass  # leave as NaN

    t2_map = np.full(mask.shape, np.nan)
    t2_map[mask] = t2_fit
    return t2_map

def process_T2_map(t2_paths: tuple, vials_config: list):
    print("Processing T2 mapping data")
    stack = load_t2_stack(t2_paths, vials_config)
    t2_map = fit_t2(stack)
    results = summarise_map(t2_map, stack["labels"], vials_config, "T2")
    comparison = compare_to_standard(results, THERMOMETRY_RESULTS_PATH, STANDARD_DATA_PATHS, "T2")
    out_path = t2_paths[1].parent / "T2_results.csv"
    comparison.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Saved T2 results and comparison to {out_path}")
    return comparison

# ====== T2 Mapping END ======

# ====== Diffusion Mapping START ======

def write_bval_bvec(bvals: np.ndarray, bvecs: np.ndarray, prefix: Path):
    """Save FSL-format .bval (one row) and .bvec (three rows) files next to `prefix`."""
    bval_path = Path(f"{prefix}.bval")
    bvec_path = Path(f"{prefix}.bvec")
    np.savetxt(bval_path, bvals[None, :], fmt="%.2f")
    np.savetxt(bvec_path, bvecs.T, fmt="%.6f")
    return bval_path, bvec_path

def load_dti_stack(paths: tuple, vials_config: list):
    """Parse b-values/directions, save .bval/.bvec, motion-correct the DWIs and mask to vials with D available.

    Returns a dict with the motion-corrected data, gradient table, 3D mask and 3D label map.
    """
    image_path, atlas_path, par_path = paths
    print(f"Parsing b-values and gradient directions from {par_path.name}")
    bvals, bvecs = extract_diffusion_info(par_path)
    prefix = atlas_path.parent / image_path.name.replace("_rigid.nii.gz", "")
    bval_path, bvec_path = write_bval_bvec(bvals, bvecs, prefix)
    print(f"Saved {bval_path.name} and {bvec_path.name}")

    # drop derived volumes (b > 0 with no direction, e.g. the Philips isotropic/trace image)
    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    keep = (bvals < 50) | (np.linalg.norm(bvecs, axis=1) > 0.5)
    print(f"Keeping {int(keep.sum())} of {len(keep)} volumes (b-values: {bvals[keep].tolist()})")
    gtab = gradient_table(bvals[keep], bvecs=bvecs[keep])

    img = nib.load(image_path)
    data = img.get_fdata(dtype=np.float64)[..., keep]
    atlas = np.rint(nib.load(atlas_path).get_fdata()).astype(int)
    labels = atlas[..., 0] if atlas.ndim == 4 else atlas
    if atlas.ndim == 4:
        labels = np.where((atlas[..., keep] == labels[..., None]).all(axis=-1), labels, 0)

    print("Motion correcting DWI volumes (dipy.align.motion_correction)")
    logging.getLogger("dipy").setLevel(logging.WARNING)  # dipy's info messages contain characters the Windows console cannot print
    corrected, _ = motion_correction(data, gtab, affine=img.affine)
    data = corrected.get_fdata()

    segment_indices = [v.segment_index for v in vials_config
                       if "D" in v.si_traceable_characterisation_available]
    print(f"Building mask from atlas: {len(segment_indices)} vials with D available "
          f"(segment indices {sorted(segment_indices)})")
    labels = np.where(np.isin(labels, segment_indices), labels, 0)
    labels = refine_vial_labels(labels, slice_range=None)
    mask = labels > 0
    print(f"Mask keeps {int(mask.sum())} of {mask.size} voxels in {len(np.unique(labels[mask]))} vials")
    return {"data": data, "gtab": gtab, "mask": mask, "labels": labels}

def fit_dti(stack: dict) -> np.ndarray:
    """DTI fit inside the mask; returns the mean diffusivity map in mm^2/s (NaN outside the mask)."""
    print("Fitting DTI model (dipy)")
    fit = TensorModel(stack["gtab"]).fit(stack["data"], mask=stack["mask"])
    return np.where(stack["mask"], fit.md, np.nan)

def process_diffusion(dti_paths: tuple, vials_config: list):
    print("Processing diffusion data")
    stack = load_dti_stack(dti_paths, vials_config)
    md_map = fit_dti(stack)
    unit = "e-3_mm2_s"  # same units as "D reported" in the standard data (10^-3 mm^2/s)
    results = summarise_map(md_map, stack["labels"], vials_config, "D", scale=1e3, unit=unit)
    comparison = compare_to_standard(results, THERMOMETRY_RESULTS_PATH, STANDARD_DATA_PATHS, "D", unit=unit)
    out_path = dti_paths[1].parent / "diffusion_results.csv"
    comparison.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Saved diffusion results and comparison to {out_path}")
    return comparison

# ====== Diffusion Mapping END ======

# ====== Thermometry START ======

def load_echo_times(par_path: Path):
    """Read echo times from the PAR file and return them as a numpy array in seconds."""
    return extract_echo_times(par_path) * 1e-3  # convert from ms to s

def masked_mean_per_echo(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of data within the mask for each echo."""
    return (data * mask).sum(axis=(0, 1, 2)) / mask.sum(axis=(0, 1, 2))

def process_thermo(thermo: tuple, thermo_shifted: tuple, vials_config: list):
    print(f"Processing thermometry data: {thermo[0].name} and {thermo_shifted[0].name}")
    all_thermo = np.concatenate([
        nib.load(thermo[0]).get_fdata(dtype=np.float64),
        nib.load(thermo_shifted[0]).get_fdata(dtype=np.float64)
    ], axis=3)
    all_echo_times = np.concatenate([
        load_echo_times(thermo[2]),
        load_echo_times(thermo_shifted[2])
    ])
    sort_idcs = np.argsort(all_echo_times)
    all_thermo = all_thermo[..., sort_idcs]
    all_echo_times = all_echo_times[sort_idcs]
    mask_U = np.repeat(get_mask(thermo[1], vials_config, 'U'), 2, axis=-1)
    mask_V = np.repeat(get_mask(thermo[1], vials_config, 'V'), 2, axis=-1)
    mean_U = masked_mean_per_echo(all_thermo, mask_U)
    mean_V = masked_mean_per_echo(all_thermo, mask_V)
    results = {}
    for vial_id, signal in (('U', mean_U), ('V', mean_V)):
        results[vial_id] = fit_vial(all_echo_times, signal)
    df_out = pd.DataFrame(
        [
            {
                "vial": vial_id,
                "df_Hz": r["df"],
                "temperature_C": r["temperature"],
                "r_squared": r["r_squared"],
            }
            for vial_id, r in results.items()
        ]
    )
    out_path = Path(thermo[0]).parent / "thermometry_results.csv"
    df_out.to_csv(out_path, index=False, float_format="%.4f")
    return results

def fit_vial(echo_times: np.ndarray, signal: np.ndarray) -> dict:
    """Fit the dual-resonance model to a region-mean signal and estimate temperature."""
    a0 = float(np.max(signal)) / 2
    initial_guess = [a0, a0, 20.0, 20.0, 217.0, 0.0]  # A1, A2, R2*_1, R2*_2, df (Hz), dphi (deg)
    popt, pcov, r_squared = lsq_fit_thermometry_signal_model(echo_times, signal, initial_guess)
    df = popt[4]
    return {
        'popt': popt,
        'pcov': pcov,
        'r_squared': r_squared,
        'df': df,
        'temperature': float(calculate_temperature_from_df(df, FIELD_STRENGTH)),
    }

# ====== Thermometry END ======

# ====== Diffusion START ======



# ====== Diffusion END ======

#For full QA use "SPIRIT dev 20261005", for thermometry use "SPIRIT dev thermo 20261006"
run(Path(R"SPIRIT dev 20261005"))