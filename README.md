# acme-emulator
"Another Cosmological halo Model Emulator" (ACME)

## Installation

Clone this git repository. It is recommended to create a new virtual environment for ACME (Python 3.7+ is *required* for using ACME). Then install all required Python packages as such:
```
conda create -n acme -f acme.yaml
conda activate acme
```

You will need to compile cHOD and fastcorrelation as such:
```
cd cHOD
make
cd ../fastcorrelation
make
```

In order to compute parameter posterior distributions as done in the ACME paper, you need to download and compile the MultiNest library [https://github.com/farhanferoz/MultiNest]. (The path to the libmultinest.so file must also be added to your LD_LIBRARY_PATH.) If you only wish to use the emulator code itself, this is not necessary.

## Usage

You will need to download the simulation data products by running:
```
python download_sim_files.py
```

*This will require approximately 250 GB of storage and it may take 24 hours or more to download, even over a fast connection.* **This is only required to re-create the emulator 'from scratch.'** If you only want to use the precomputed emulator as described in Wibking+ (2019), this is not necessary.

You can specify another download location (e.g. on another filesystem) with the --download_path option. The analysis scripts assume that the simulation files are accessible from within this directory, so you will need to create a symbolic link to the download path from the repository directory.

### Converting file formats

```
doit -f convert_abacuscosmos.py sample=lowz
```

### Creating the Latin Hypercube

The Latin Hypercube design for the training data can be created via:
```
python create_params_lowz_emu.py
```
*This step is **not** deterministic. If you delete the created parameter files, there's no way to re-generate them.* This may be fixed in a future version.

### Computing the training data

The training data can be computed with:
```
doit -f analyze_abacuscosmos.py sample=lowz
```

If you are reproducing the training data used in Wibking+ (2019), this step may take 5-7 days or longer, depending on your CPU and storage speed.

### Training the emulator

The emulator for projected galaxy clustering ($w_p$) can be trained by
```
python Analysis/train_emu.py Params/LOWZ_HOD/NHOD_lowz.*.seed_42.template_param
```

For approximately 400 training points, this will take 30 seconds to 5 minutes per correlation function bin, depending on your CPU speed and whether you modify the optimization parameters.



