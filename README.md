1. use `parrec_to_nii.ipynb` to turn high res T1 from par/rec to nifti
2. run `spirit-phantom register high-res-T1.nii.gz`
3. move `registered_data` into `SPIRIT dev * 2026100*`
4. run `preprocess.py`
    a. extract echo times for thermometry from PAR
    b. resample atlas to the imaged FoV for each scan
    c. rigidly register 4D image stacks along echo train
5. 