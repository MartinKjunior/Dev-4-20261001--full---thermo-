from __future__ import annotations

import traceback
from pathlib import Path

import nibabel as nib

from register_rigid_elastix import register_rigid
from resample_atlas import resample_atlas

DATA_DIR = Path(__file__).parent / "SPIRIT dev thermo 20261006"
OUT_DIR = DATA_DIR / "processed"


def find_atlas(data_dir: Path) -> Path:
    """Find the atlas in the timestamped subfolder (the folder name can change)."""
    atlases = sorted(data_dir.glob("**/transformed_component_atlas.nii.gz"))
    if not atlases:
        raise FileNotFoundError(f"No transformed_component_atlas.nii.gz found in {data_dir}\\*")
    if len(atlases) > 1:
        print(f"Multiple atlases found, using the latest: {atlases[-1]}")
    return atlases[-1]


def main() -> None:
    atlas_path = find_atlas(DATA_DIR)
    rec_files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".rec")
    OUT_DIR.mkdir(exist_ok=True)

    for rec in rec_files:
        print(f"\n{rec.name}")
        if "Survey" in rec.name:
            print("  skipped: survey")
            continue
        try:
            ndim = nib.load(rec).ndim
            if ndim not in (3, 4):
                print(f"  skipped: {ndim}D image not supported")
                continue

            atlas = resample_atlas(atlas_path, rec)
            out = OUT_DIR / f"{rec.stem}_atlas.nii.gz"
            if out.exists():
                print(f"  skipped: {out.name} already exists")
                continue
            nib.save(atlas, out)

            if ndim == 4:
                nib.save(register_rigid(rec), OUT_DIR / f"{rec.stem}_rigid.nii.gz")
        except Exception:
            print(f"  FAILED:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
