from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Definition lines look like "#  echo_time    (float)" or "#  pixel spacing (x,y) (in mm)    (2*float)"
DEFINITION_LINE = re.compile(r"^#\s+(?P<name>.+?)\s+\((?:(?P<n>\d+)\*)?(?:integer|float|string)\)\s*$")


def extract_echo_times(par_path: str | Path) -> np.ndarray:
    """Return the echo time (ms) of every row of the PAR IMAGE INFORMATION table.

    The echo_time column is located by counting the fields listed in the
    IMAGE INFORMATION DEFINITION section (an entry such as '2*float' occupies
    two columns). Values are returned in table order as a 1D float array.
    """
    par_path = Path(par_path)
    echo_times = []
    in_definition = False
    in_table = False
    column = 0  # running column count while reading the definition
    echo_column = None

    with open(par_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("# === IMAGE INFORMATION DEFINITION"):
                in_definition = True
                continue
            if stripped.startswith("# === IMAGE INFORMATION ="):
                if echo_column is None:
                    raise ValueError("echo_time not found in the IMAGE INFORMATION DEFINITION")
                in_definition, in_table = False, True
                continue

            if in_definition:
                m = DEFINITION_LINE.match(stripped)
                if m:
                    if m.group("name").startswith("echo_time"):
                        echo_column = column
                    column += int(m.group("n") or 1)
            elif in_table:
                if not stripped or stripped.startswith("#"):
                    continue  # blank lines and the column-title comment line
                echo_times.append(float(stripped.split()[echo_column]))

    return np.array(echo_times, dtype=np.float64)


def extract_diffusion_info(par_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return per-volume b-values (s/mm^2, shape (n,)) and gradient directions (shape (n, 3)) from a PAR file.

    Volumes follow the REC order: a volume is a unique (diffusion b value number, gradient orientation number)
    combination, taken from its first image row. Philips lists directions as (ap, fh, rl); they are returned
    reordered as (rl, ap, fh), matching the voxel axes (L, P, S) of the nibabel-loaded image.
    """
    par_path = Path(par_path)
    names, columns = [], {}
    rows = []
    in_definition = in_table = False
    column = 0
    with open(par_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("# === IMAGE INFORMATION DEFINITION"):
                in_definition = True
                continue
            if stripped.startswith("# === IMAGE INFORMATION ="):
                in_definition, in_table = False, True
                continue
            if in_definition:
                m = DEFINITION_LINE.match(stripped)
                if m:
                    columns[m.group("name")] = column
                    column += int(m.group("n") or 1)
            elif in_table and stripped and not stripped.startswith("#"):
                rows.append(stripped.split())

    def col(prefix: str) -> int:
        for name, idx in columns.items():
            if name.startswith(prefix):
                return idx
        raise ValueError(f"'{prefix}' not found in the IMAGE INFORMATION DEFINITION of {par_path}")

    c_idx, c_b, c_bnum, c_grad, c_dir = (col("index in REC file"), col("diffusion_b_factor"),
                                         col("diffusion b value number"), col("gradient orientation number"),
                                         col("diffusion (ap, fh, rl)"))
    rows.sort(key=lambda r: int(r[c_idx]))
    volumes = {}
    for r in rows:
        volumes.setdefault((int(r[c_bnum]), int(r[c_grad])), r)
    bvals = np.array([float(r[c_b]) for r in volumes.values()])
    ap_fh_rl = np.array([[float(v) for v in r[c_dir:c_dir + 3]] for r in volumes.values()])
    return bvals, ap_fh_rl[:, [2, 0, 1]]
