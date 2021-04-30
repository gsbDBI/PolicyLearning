<h1 align="center">Policy Learniing with Adaptively Collected Data</h1>

Models for paper _Policy Learniing with Adaptively Collected Data_.

<p align="center">
  Table of contents </br>
  <a href="#overview">Overview</a> •
  <a href="#development-setup">Development Setup</a> •
  <a href="#quickstart-with-model">Quickstart</a> 
</p>


# Overview

*Note: For any questions, please file an issue.*

Adaptive experimental designs can dramatically improve efficiency in randomized trials. But adaptivity also makes offline policy inference challenging. In the paper _Off-Policy Evaluation via Adaptive Weighting with Data from Contextual Bandits_, we propose a class of estimators that lead to asymptotically normal and consistent policy evaluation. This repo contains reproducible code for the results shown in the paper. 

We organize the code into two directories:
- [./utils](https://github.com/gsbDBI/PolicyLearning/tree/main/utils) is a Python module for policy learning using generalized AIPW estimator, as described in the paper.

- [./scripts](https://github.com/gsbDBI/PolicyLearning/tree/main/scripts) contains python scripts to run experiments and make plots shown in the paper, including:
   - collecting contextual bandits data with a floored Thompson sampling agent;
   - doing off-line policy learning using collected data;
   - saving results and making plots. 

# Development setup
R and Python are required. We recommend creating the following conda environment for computation.
```bash
conda create --name PolicyLearning python=3.7
conda activate policy_learning
source install.sh
```

# Quickstart with model

- To do policy learning and reproduce results shown in the paper, please follow the instructions in [./scripts/README.md](https://github.com/gsbDBI/PolicyLearning/tree/main/scripts/README.md).
- For a quick start on one simulation using synthetic data of sample size 1000 , use
```bash
source activate policy_learning
cd ./experiments/
python script_synthetic.py -s 1 -n test
```
Results will be saved in ./experiments/results/


