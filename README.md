# EvalRS-CIKM-2022

## Overview

This is the solution presented in the "Triplet losses-based matrix factorization for robust recommendations" paper, for the EvalRS challenge @ CIKM 2022. 

While this README is currently a stub, it will contain a more detailed description of the solution later on. 

## Instructions

This solution requires Python 3.x to be executed. The following are the steps to reproduce the results.

1. (optional) Create a virtual environment to install everything needed. If not created, all libraries (see next step) will be installed system-wide.
```bash
virtualenv venv
. venv/bin/activate
```

2. Install the dependencies in requirements.txt
```bash
pip install -r requirements.txt
```
Note that, although needed, PyTorch does not show up among the dependencies. This is because its installation is system-dependent (in particular for what concerns CUDA). If not already available, you may follow the instructions [here](https://pytorch.org/get-started/locally/).

3. Create the upload.env file, with all of the relavant information (see local.env for a list of required fields)

4. Run the code, through submission.py
```bash
python submission.py
```

## Troubleshooting
