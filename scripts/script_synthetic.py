from utils.saving import *
import matplotlib.pyplot as plt
from utils.policy_tree import fit_policytree, predict_policytree
from utils.datagen import *
from utils.experiment import *
from utils.inference import aw_scores
from utils.thompson import draw_thompson
from time import time
import pickle
import os
from random import choice
import argparse
from itertools import product
import sys
from utils.ridge import *


parser = argparse.ArgumentParser(description='Process DGP settings')
parser.add_argument(
    '-n',
    '--name',
    type=str,
    default='synthetic',
    help='saving name of experiments')
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
    '--split',
    type=float,
    default=0.5,
    help='split')
parser.add_argument(
    '--noise_std',
    type=float,
    default=1.0,
    help='noise std')
parser.add_argument(
    '-B',
    type=float,
    default=1.5,
    help='covariate sampling Uniform[-B, B].')
parser.add_argument(
    '-b',
    '--bandit_model',
    type=str,
    default='TSModel',
    help='bandit model')
parser.add_argument(
    '-m',
    '--muhat',
    type=str,
    default='lfo',
    help='muhat model, lfo or lco')
parser.add_argument(
    '-s',
    '--sims',
    type=int,
    default=150,
    help='number of simulations')
parser.add_argument(
    '-T',
    '--T',
    type=int,
    default=20000,
    help='sample size')
parser.add_argument(
    '-p',
    '--policy_class',
    type=str,
    default='tree',
    help='policy class')


if __name__ == '__main__':

    t1 = time()

    """ Experiment configuration """
    project_dir = os.getcwd()
    args = parser.parse_args()

    num_sims = args.sims
    save_every = 10

    # configs
    K = 2
    p = 3
    T = args.T
    recorded_T = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    recorded_T = [T]
    mask = np.zeros((T, T))
    for i in range(T):
        mask[i, i:] = 1
    batch_size = 100
    time_explore = 50 * K
    batch_sizes = [time_explore] + [batch_size] * \
        int((T - time_explore)/batch_size)
    if np.sum(batch_sizes) < T:
        batch_sizes[-1] += T - np.sum(batch_sizes)

    """ Generate test data """
    T_test = 100000
    noise_std = 1.0
    data_test, _ = one_dim_data(
        T=T_test, K=K, p=p, B=args.B, noise_std=args.noise_std, signal=args.signal)

    """ Run and analyze the experiment """
    regret = []
    for s in range(num_sims):
        """ Data generation """
        # Collect data from environment
        data_exp, mus = one_dim_data(
            T=T, K=K, p=p, B=args.B, noise_std=args.noise_std, signal=args.signal)
        xs, ys = data_exp['xs'], data_exp['ys']

        # Run the experiment on the simulated data
        data = run_experiment(xs, ys, floor_start=1/K, floor_decay=args.floor_decay,
                              bandit_model=args.bandit_model, batch_sizes=batch_sizes,
                              recorded_T=recorded_T)
        yobs, ws, probs = data['yobs'], data['ws'], data['probs']
        muhat = ridge_muhat_lfo_pai(xs, ws, yobs, K, batch_sizes)
        balwts = 1 / collect(collect3(probs), ws)
        gammahat = aw_scores(yobs=yobs, ws=ws, balwts=balwts,
                             K=K, muhat=muhat)

        for TT, recorded_agent in zip(recorded_T, data['recorded_agents']):
            # Estimate muhat and gammahat
            def learn_policy(weighting):
                if weighting == 'uniform':
                    policy = fit_policytree(xs[:TT], gammahat[:TT])
                    w_test = predict_policytree(
                        policy, data_test['xs'])
                elif weighting.startswith('alpha'):
                    beta = float(weighting.split(
                        '_')[-1]) * args.floor_decay
                    weights = 1.0/np.arange(1, TT+1)**beta
                    policy = fit_policytree(
                        xs[:TT], gammahat[:TT], weights=weights)
                    w_test = predict_policytree(
                        policy, data_test['xs'])
                policy_value = np.mean(
                    data_test['muxs'][np.arange(T_test), list(w_test)])
                return policy_value, w_test

            policy_value_uniform, arm_uniform = learn_policy('uniform')
            beta_rate = [0.5, 1.0, 1.5, 2.0]
            policy_regret_beta = []
            policy_value = args.B ** 2 / 3 - 1 + 4 / (3 * args.B)
            for b_rate in beta_rate:
                p_v, _ = learn_policy(f'alpha_{b_rate}')
                policy_regret_beta.append(policy_value * args.signal-p_v)

            # calculate agent out_of_sample regret
            w_agent, _ = draw_thompson(data_test['xs'], recorded_agent,
                                       0, T_test, 0, floor=0)
            policy_value_agent = np.mean(
                data_test['muxs'][np.arange(T_test), w_agent])

            # record results
            regret.append([TT, args.B,
                           args.signal, args.floor_decay,
                           args.policy_class,
                           args.signal * policy_value - policy_value_uniform,
                           *policy_regret_beta,
                           args.signal*policy_value-policy_value_agent])

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
