# acme-emulator
"Another Cosmological halo Model Emulator" (ACME)

Source files and functions are individually documented *(currently incomplete)* here: https://benwibking.github.io/acme-emulator/


## Installation

Clone this git repository. It is recommended to create a new virtual environment for ACME (Python 3.7+ is *required* for using ACME). Then install all required Python packages as such:
```
git clone https://github.com/BenWibking/acme-emulator.git
cd acme-emulator
conda env create -n acme --file acme.yml
conda activate acme
```

In order to compute *new* training data, you will need to compile cHOD and fastcorrelation as such:
```
cd cHOD
make
cd ../fastcorrelation
make
```

In order to compute parameter posterior distributions as done in the ACME paper, you need to download and compile the MultiNest library [https://github.com/farhanferoz/MultiNest]. (The path to the libmultinest.so file must also be added to your LD_LIBRARY_PATH.) If you only wish to use the emulator code itself, this is not necessary.


## Usage

If you want to compute new training samples (or to re-compute the training samples used in Wibking+ (2019) from scratch) You will need to download the *reduced* simulation data products by running:
```
python download_reduced_sims.py
```

*This will require 20.6 GB of storage and it may take several hours to download.* **This is only required to re-create or extend the emulator.** If you only want to use the precomputed emulator as described in Wibking+ (2019), this is not necessary.

The files will be downloaded to the ./AbacusCosmos/ directory.  If the download script is interrupted or killed, you can re-run the script and the download will automatically restart where it left off.


### Creating a Latin Hypercube

If you only want to re-create the training data used in Wibking+ (2019), do not perform this step. The necessary parameter files are already included in the ./Params/LOWZ_HOD_precomputed directory. Modify subsequent commands in this document to use this directory instead.

However, if you wish to create a Latin Hypercube design for *new* training data, run the commands:
```
mkdir Params/LOWZ_HOD
python create_params_lowz_emu.py
```
*This step is **not** deterministic. If you delete the created parameter files, there's no way to re-generate them.* This may be fixed in a future version.


### Computing the training data

If you only want to use the training data already computed in Wibking+ (2019), do not perform this step. The necessary training data are already included in the ./Params/LOWZ_HOD_precomputed directory. Skip ahead to 'training the emulator' below.

The training data can be (re-)computed with:
```
doit -n 4 -f analyze_abacuscosmos.py sample=lowz
```
where `-n 4` specifies that four processes will run simultaneously. This number should be less than or equal to the number of CPU cores on the machine you are running on.

If you are re-computing the training data used in Wibking+ (2019), this step may take 5-7 days or longer, depending on your CPU and storage speed.

You will need to run this commmand a second time to finish the computation. (This may be fixed in a future version.)
```
doit -f analyze_abacuscosmos.py sample=lowz
```


### Training the emulator

The emulator for projected galaxy clustering w_p can be (re-)trained by running:
```
python Analysis/train_emu.py Params/LOWZ_HOD_precomputed/*.seed_42.template_param
```

For approximately 400 training points, this will take 30 seconds to 5 minutes per correlation function bin, depending on your CPU speed.


### Computing auxiliary data

In order to use the emulator to draw inferences from data, it is strongly advised that you also compute the following auxiliary pieces of data that account for sample variance and residual redshift space distortions, which are effects not included in the training data.

#### Computing correction from ensemble average

To be written. Precomputed version is ./Params/lowz_wp_correction.txt

#### Computing (fiducial) RSD correction

To be written. Precomputed version is ./Params/lowz_rsd_correction.txt



### Computing predictions from the emulator

See `Analysis/predict_emulator.py` for an example of how to use the emulator to make predictions as a function of HOD and cosmological parameters:
```
python Analysis/predict_emulator.py Params/NHOD_lowz_fiducial_params.template_param fiducial_wp.txt
```


### Computing posterior samples with MultiNest

See `Analysis/fit_clustering.py` for an example of how to do this. Covariance matrices and data are not currently provided.
```
usage: fit_clustering.py [-h] [--multinest MULTINEST] [--mcmc]
                         [--mcmc-ppd MCMC_PPD] [--mcmc-chain MCMC_CHAIN]
                         [--wp-correction WP_CORRECTION]
                         [--ds-correction DS_CORRECTION]
                         [--rsd-correction RSD_CORRECTION]
                         [--cov-inv-output COV_INV_OUTPUT]
                         [--cov-output COV_OUTPUT]
                         wp_obs_file input_wp_emu_filename wp_cov_file
                         ds_obs_file input_ds_emu_filename ds_cov_file
                         fiducial_param_file param_sampling_file fit_wp_file
                         fit_ds_file fit_param_file
fit_clustering.py: error: the following arguments are required: wp_obs_file, input_wp_emu_filename, wp_cov_file, ds_obs_file, input_ds_emu_filename, ds_cov_file, fiducial_param_file, param_sampling_file, fit_wp_file, fit_ds_file, fit_param_file
```