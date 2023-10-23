# JLab Data Science Toolkit

Composible elements for data science workflows

## Software Requirement

- Python 3.10
- Python packages are defined in the yaml and setup.py
- This document assumes you are running at the top directory

## Installing 
* Pull code from repo
```
git clone https://github.com/JeffersonLab/jlab_datascience_core.git
```
* Create default conda environment setup (for mac M1):
```
conda env create --file tf-metal-arm64.yaml (only once)
conda activate tf-metal-evms-ultra (required every time you use the package)
```
* Install repo package in environment:
```
cd jlab_datascience_core
pip install -e . --user
```

## Directory Organization
```
├── setup.py
├── datascience_toolkit
    ├── core                              : folder containing base classes
    ├── dataprep                          : a folder with code to read and prep data
    ├── models                            : a folder with code to for models
    ├── cfg                               : a folder contains configuration
    ├── utils                             : a folder contains utilities
          
```
