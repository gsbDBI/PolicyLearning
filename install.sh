#!/usr/bin/env bash
python setup.py develop
conda install -c conda-forge rpy2 openml jupyter
conda install -c conda-forge r-devtools r-grf r-bh

