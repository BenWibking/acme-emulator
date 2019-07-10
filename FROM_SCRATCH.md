# acme-emulator
"Another Cosmological halo Model Emulator" (ACME)

This file describes how to re-create the emulator from the original AbacusCosmos simulation data release. Note that this is *not* necessary for normal use (see README.md).


## Downloading original raw simulation files

You can download the original AbacusCosmos simulation data products by running:
```
python download_sim_files.py
```

*WARNING: This will require approximately 350 GB of storage and it may take 24-48 hours or more to download, even over a fast connection.*

You can specify another download location (e.g. on another filesystem) with the --download_path option. The analysis scripts assume that the simulation files are accessible from within this directory, so you will need to create a symbolic link to the download path from the repository directory.


### Converting file formats

This reduces storage requirements, since we only keep a subset of halo properties and a subsample of the simulation particles. The scripts in Conversion/ can be modified to include additional halo properties from the original Rockstar halo catalogs if you need them for your custom analysis.
```
doit -f convert_abacuscosmos.py sample=lowz
```

This step is time consuming (~ 2 days). After this completes, you can delete the original downloaded files from step 1 if you wish to reclaim storage space.


### Creating the Latin Hypercube

A new Latin Hypercube design for the training data can be created via:
```
mkdir Params/LOWZ_HOD
python create_params_lowz_emu.py
```
*This step is **not** deterministic. If you delete the created parameter files, there's no way to re-generate them.* This may be fixed in a future version.

If you wish to re-create the emulator as in Wibking+ (2019), do not perform this step. Use the *.template_param files included in ./Params/LOWZ_HOD_precomputed/ and copy them to ./Params/LOWZ_HOD/.


### Computing the training data

The training data can be computed with:
```
doit -f analyze_abacuscosmos.py sample=lowz
```

For training data of the same size as used in Wibking+ (2019), this step may take 5-7 days or longer, depending on your CPU and storage speed.

You will need to run this commmand a second time to finish the computation. (This may be fixed in a future version.)
```
doit -f analyze_abacuscosmos.py sample=lowz
```


### Training the emulator

The emulator for projected galaxy clustering ($w_p$) can be trained by
```
python Analysis/train_emu.py Params/LOWZ_HOD/NHOD_lowz.*.seed_42.template_param
```

For approximately 400 training points, this will take 30 seconds to 5 minutes per correlation function bin, depending on your CPU speed and whether you modify the optimization parameters.



