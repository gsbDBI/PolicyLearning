This directory contains scripts to run experiments using synthetic data and public classification datasets on [OpenML](https://www.openml.org), 
and make plots shown in the paper _Policy Learning with Adaptively Collected Data_.

## File description
- `script_synthetic.py`: run simulations on synthetic data described in the paper, and save results in `./results/` computed by different weighting schemes.
- `script_classification.py`: run simulations on classification data from OpenML, and save results in `./results/` computed by different weighting schemes.
- `plots_synthetic.ipynb`: make plots of performances on synthetic data shown in the paper using saved results in `./results/`.
- `plots_classification.ipynb`: make plots abd tables of performances on OpenML data shown in the paper using saved results in `./results/`.


## Reproducibility 
To reproduce results on synthetic data shown in the paper, do
1. Run DGP with signal: `python script_synthetic.py -s 1000` to run experiments and save results in `./results/`.
2. Open `plots_synthetic.ipynb`, follow the instructions in the notebook to generate plots based on the saved results in `./results/`. 

To reproduce results on classification datasets shown in the paper, do
1. `python script_classification.py -s 100 -f {NameOfDataset}` to run experiments and save results in `./results/`.
2. Open `plots_classification.ipynb`, follow the instructions in the notebook to generate plots based on the saved results in `./results/`. 


## Quick start for running simulations using synthetic data
Load packages
```python
from utils.saving import *
from utils.policy_tree import fit_policytree, predict_policytree
from utils.datagen import *
from utils.experiment import *
from utils.inference import aw_scores
from utils.thompson import draw_thompson
from utils.ridge import *
```
### 1. Collecting contextual bandits data
```python
K = 2 # Number of arms
p = 3 # Number of features
T = 20000 # Sample size
B = 1.5 # Covariates are sampled uniformly from [-B, B]^p
batch_sizes = [100] * 200 # Batch sizes

# Collect data from environment
data_exp, mus = one_dim_data(
       T=T, K=K, p=p, B=B, noise_std=1.0, signal=1.0)
xs, ys = data_exp['xs'], data_exp['ys']
data = run_experiment(xs, ys, floor_start=1/K, floor_decay=0.5,
                              bandit_model='TSModel', batch_sizes=batch_sizes,
                              recorded_T=[T])
yobs, ws, probs = data['yobs'], data['ws'], data['probs']
```

### 2. Policy learning with uniform weights
```python
# Estimate muhat and gammahat
muhat = ridge_muhat_lfo_pai(xs, ws, yobs, K, batch_sizes)
gammahat = aw_scores(yobs=yobs, ws=ws, balwts=1 / collect(collect3(probs), ws),
                     K=K, muhat=muhat)
policy = fit_policytree(xs, gammahat)
# simulate standalone test data
data_test, _ = one_dim_data(
        T=T_test, K=K, p=p, B=B, noise_std=1.0, signal=1.0)
w_test = predict_policytree(policy, data_test['xs'])
policy_value = np.mean(data_test['muxs'][np.arange(T_test), list(w_test)])
optimal_policy_value = B ** 2 / 3 - 1 + 4 / (3 * B)
print(f"Regret is {optimal_policy_value-policy_value}").
```


