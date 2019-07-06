# acme-emulator
"Another Cosmological halo Model Emulator" (ACME)

## Installation

Clone this git repository. It is recommended to create a new virtual environment for ACME (Python 3.7+ is *required* for using ACME). Then install all required Python packages:
```
conda create -n acme python=3.7
conda activate acme
pip install -r requirements.txt
```

If pycurl does not install correctly using the above method, try this instead (I don't know why this happens):
```
conda install pycurl
```

## Usage

You will need to download the simulation data products by running:
```
python download_sim_files.py
```

*This will require approximately 500 GB of storage and it may take 24 hours or more to download, even over a fast connection.*

You can specify another download location (e.g. on another filesystem) with the --download_path option. The analysis scripts assume that the simulation files are accessible from within this directory, so you will need to create a symbolic link to the download path from the repository directory.

Each of the doit pipelines can be run with, e.g.
```
doit -f Workflows/analyze_abacuscosmos.py
```


