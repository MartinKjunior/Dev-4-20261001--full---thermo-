1. use `parrec_to_nii.ipynb` to turn high res T1 from par/rec to nifti
2. run `spirit-phantom register high-res-T1.nii.gz`
3. move `registered_data` into `SPIRIT dev * 2026100*`
4. run `preprocess.py`
    a. resample atlas to the imaged FoV for each scan
    b. rigidly register 4D image stacks along echo train
5. run `pipeline.py`
    - I made a mistake recording thermometry data the first time so it's in a different folder, need to change the path at the very bottom of this file to run relaxometry/diffusion or thermometry

## What this workspace does

Quality-assurance processing for the SPIRIT diffusion QA phantom (Gold Standard Phantoms). Phantom scans (Philips PAR/REC) are segmented into vials using an atlas, quantitative maps are computed, and the mean value in each vial is compared against the temperature-dependent reference ("standard") data supplied with the phantom.

### Files

- `parrec_to_nii.ipynb`: converts PAR/REC to NIfTI.
- `resample_atlas.py`, `register_rigid_elastix.py`, `preprocess.py`: resample the atlas to each scan's FoV and rigidly register 4D stacks (outputs go in each scan's `processed` folder).
- `par_parser.py`: reads PAR headers (repetition time, echo times, b-values and gradient directions).
- `pipeline.py`: main processing, see below.
- `thermometry.py`: temperature estimation helpers; `temperature.txt` holds temperature notes.
- `ground_truth/`: standard data spreadsheets (T1, T2, D vs temperature per product).
- `dev.ipynb`: exploratory plots of the standard data against temperature (quadratic fit per product).
- `LICENSE`: MIT.

### What `pipeline.py` computes

- **Thermometry**: temperature from the echo-shift gradient-echo scans, written to `thermometry_results.csv`. The mean temperature is used for the comparisons below.
- **T1**: variable flip angle (sT1W 3D FFE) data; a linear fit gives initial values for a non-linear fit per voxel. Output `T1_results.csv`.
- **T2**: multi-echo GRASE data, mono-exponential fit (linear initial guess, then non-linear). Output `T2_results.csv`.
- **Diffusion**: DTI data; b-values and gradient directions are parsed from the PAR file and saved as `.bval`/`.bvec`, motion-corrected and fitted with dipy (mean diffusivity). Output `diffusion_results.csv`.

For T1, T2 and diffusion, only voxels inside vials that have the relevant characterisation available are fitted. Each result table gives the per-vial mean and standard deviation, the expected value from a quadratic fit of the standard data against temperature (evaluated at the measured mean temperature, with uncertainty), and a p-value for the measurement against the expected value.