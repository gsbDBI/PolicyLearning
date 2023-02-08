#!/usr/bin/env bash
python setup.py develop
conda install -c anaconda jupyter
conda install -c conda-forge rpy2 openml
conda install -c conda-forge r-devtools r-grf r-bh

