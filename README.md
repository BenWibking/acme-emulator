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

Each of the doit pipelines can be run with, e.g.
```
doit -f Workflows/analyze_abacuscosmos.py
```


