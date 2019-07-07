# acme-emulator
"Another Cosmological halo Model Emulator" (ACME)

## Installation

Clone this git repository. It is recommended to create a new virtual environment for ACME (Python 3.7+ is *required* for using ACME). Then install all required Python packages as such:
```
conda create -n acme python=3.7
conda activate acme
conda install pycurl
pip install -r requirements.txt
```

In order to compute parameter posterior distributions as done in the ACME paper, you need to download and compile the MultiNest library [https://github.com/farhanferoz/MultiNest]. If you only wish to use the emulator code itself, this is not necessary.

## Usage

You will need to download the simulation data products by running:
```
python download_sim_files.py
```

*This will require approximately 500 GB of storage and it may take 24 hours or more to download, even over a fast connection.*

You can specify another download location (e.g. on another filesystem) with the --download_path option. The analysis scripts assume that the simulation files are accessible from within this directory, so you will need to create a symbolic link to the download path from the repository directory.

### Converting file formats

```
doit -f convert_abacuscosmos.py sample=lowz
```

### Running the analysis

Each of the doit pipelines can be run with, e.g.
```
doit -f analyze_abacuscosmos.py
```


