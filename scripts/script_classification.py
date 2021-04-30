from utils.saving import *
import matplotlib.pyplot as plt
from utils.policy_tree import fit_policytree, predict_policytree
from utils.datagen import *
from utils.experiment import *
from utils.inference import aw_scores
from time import time
import pickle
import os
from random import choice
import argparse
from itertools import product
import sys
import openml


parser = argparse.ArgumentParser(description='Process DGP settings')
parser.add_argument(
    '-n',
    '--name',
    type=str,
    default='classification',
    help='saving name of experiments')
parser.add_argument(
    '--file_name',
    type=str,
    default='volcanoes-b3',
    help='name of dataset')
parser.add_argument(
    '--floor_decay',
    type=float,
    default=0.5,
    help='assignment probability floor decay')
parser.add_argument(
    '--signal',
    type=float,
    default=1.0,
    help='signal strength')
parser.add_argument(
    '--noise_std',
    type=float,
    default=1.0,
    help='noise std')
parser.add_argument(
    '-b',
    '--bandit_model',
    type=str,
    default='TSModel',
    help='bandit model')
parser.add_argument(
    '-s',
    '--sims',
    type=int,
    default=150,
    help='number of simulations')
parser.add_argument(
    '-p',
    '--policy_class',
    type=str,
    default='tree',
    help='policy class')
parser.add_argument(
    '--tree_depth',
    type=int,
    default=2,
    help='depth of policy tree.')


if __name__ == '__main__':

    t1 = time()

    """ Experiment configuration """
    project_dir = os.getcwd()
    args = parser.parse_args()

    num_sims = args.sims
    save_every = 10

    """ Load data sets """
    dname = args.file_name
    openml_list = openml.datasets.list_datasets()
    dataset = openml.datasets.get_dataset(dname)
    X, y, _, _ = dataset.get_data(dataset_format="array", target=dataset.default_target_attribute)

    """ Run and analyze the experiment """
    regret = []
    for s in range(num_sims):
        """ Data generation """
        # Collect data from environment
        data_exp, mus = generate_bandit_data(X=X, y=y, noise_std=args.noise_std, signal_strength=args.signal)
        xs, ys = data_exp['xs'], data_exp['ys']
        K, p, T = data_exp['K'], data_exp['p'], data_exp['T']
        xs_test, muxs_test, T_test = data_exp['xs_test'], data_exp['muxs_test'], data_exp['T_test']
        mask = np.zeros((T, T))
        for i in range(T):
            mask[i, i:] = 1

        batch_size = min(100, T//10)
        time_explore = batch_size // 2 * K
        batch_sizes = [time_explore] + [batch_size] * int((T - time_explore)/batch_size)
        if np.sum(batch_sizes) < T:
            batch_sizes[-1] += T - np.sum(batch_sizes)

        # Run the experiment on the simulated data
        data = run_experiment(xs, ys, floor_start=1/K, floor_decay=args.floor_decay, bandit_model=args.bandit_model, batch_sizes=batch_sizes)
        yobs, ws, probs = data['yobs'], data['ws'], data['probs']

        muhat = ridge_muhat_lfo_pai(xs, ws, yobs, K, batch_sizes)
        balwts = 1 / collect(collect3(probs), ws)
        gammahat = aw_scores(yobs=yobs, ws=ws, balwts=balwts, K=K, muhat=muhat)

        def learn_policy(weighting):
            if weighting == 'uniform':
                policy = fit_policytree(xs, gammahat, depth=args.tree_depth)
                w_test = predict_policytree(policy, xs_test)
            elif weighting.startswith('alpha'):
                beta = float(weighting.split('_')[-1]) * args.floor_decay
                weights = 1.0/np.arange(1, T+1)**beta
                policy = fit_policytree(xs, gammahat, weights=weights, depth=args.tree_depth)
                w_test = predict_policytree(policy, xs_test)
            policy_value = np.mean(muxs_test[np.arange(T_test), list(w_test)])
            return policy_value, w_test

        policy_value_uniform, arm_uniform = learn_policy('uniform')
        beta_rate = [0.5, 1.0, 2.0]
        policy_regret_beta = []
        for b_rate in beta_rate:
            p_v, _ = learn_policy(f'alpha_{b_rate}')
            policy_regret_beta.append(args.signal-p_v)

        regret.append([args.file_name, T, T_test, K, p, 
                       args.signal, args.noise_std, args.floor_decay,
                       args.policy_class, args.tree_depth, 
                       args.signal-policy_value_uniform,
                       *policy_regret_beta])
        print(regret[-1])

        if (s+1) % save_every == 0 or ((s+1) == num_sims):
            if on_sherlock():
                experiment_dir = get_sherlock_dir('policy_learning')
            else:
                experiment_dir = os.path.join(project_dir, 'results')
            if not os.path.isdir(experiment_dir):
                os.makedirs(experiment_dir)

            filename = compose_filename(args.name, 'npy')
            write_path = os.path.join(experiment_dir, filename)
            np.save(write_path, regret)
            regret = []
    print(f'Running time {time() - t1}s')
