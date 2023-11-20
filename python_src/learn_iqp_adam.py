import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrnd
import flax.linen as nn
import jax.tree_util as jtu

import sys
from pathlib import Path
import argparse
import re
import json

from utils import abs2, kl_div
from natural_gradient import ExactSampler, SecondOrderMethod, FisherAndGrad
from models import FullyConnected
from functools import partial

import optax
#import jax.example_libraries.optimizers as optimizers

from numpy_bin_tools import to_bin_array

def kl_div(p1, p2):
    return jnp.dot(p1, jnp.log(p1 / p2))

def l1_dist(p1, p2):
    return jnp.abs(p1 - p2).sum()

if __name__ == '__main__':
    # Define constants
    MINBATCH_SIZE = 1024

    # Parser
    parser = argparse.ArgumentParser(description='Learn amplitudes of the given distribution')
    parser.add_argument('--data-path', type=str, required=True, dest='data_path')
    parser.add_argument('--learning-rate', type=float, required=True, dest='learning_rate')
    parser.add_argument('--beta2', type=float, required=True, dest='beta2')
    parser.add_argument('--alpha', type=int, required=True, dest='alpha')
    parser.add_argument('--idx', type=int, required=True)
    parser.add_argument('--total-iter', type=int, required=False, dest='total_iter')

    args = parser.parse_args()

    rgx_N = re.compile('N(\d+)')

    data_path = Path(args.data_path)
    N = int(rgx_N.search(data_path.stem).group(1))

    if args.total_iter is not None:
        TOTAL_ITER = args.total_iter
    else:
        if N == 8:
            TOTAL_ITER = 20000
        elif N == 12:
            TOTAL_ITER = 40000
        elif N == 16:
            TOTAL_ITER = 80000
        elif N == 20:
            TOTAL_ITER = 160000
        elif N == 24:
            TOTAL_ITER = 320000
    RECORD_PER = TOTAL_ITER // 1000

    n_hidden = int(args.alpha)*N
    idx = int(args.idx)

    learning_rate = float(args.learning_rate)
    beta2 = float(args.beta2)

    with open('param_out.json', 'w') as param_out_f:
        param_out = vars(args)
        param_out['total_iter'] = TOTAL_ITER
        param_out['n_hidden'] = n_hidden
        json.dump(param_out, param_out_f, indent=4)

    target_probs = jnp.array(abs2(np.load(data_path)[idx,:]))
    target_sampler = ExactSampler(jnp.log(target_probs))

    key = jrnd.PRNGKey(1337)

    model = FullyConnected(n_hidden = n_hidden, dtype = jnp.float64)
    key, key_init = jrnd.split(key)
    params = model.init(key_init, jrnd.randint(key, (MINBATCH_SIZE, N), 0, 2)) # Initialization call

    fisher_and_grad = FisherAndGrad(N, model)

    params = jtu.tree_map(lambda x: x.astype(jnp.float64), params)

    optimizer = optax.adam(learning_rate = learning_rate, b2 = beta2)
    opt_state = optimizer.init(params)

    kl_divs = []
    l1_dists = []

    @partial(jax.jit, static_argnums=(0,))
    def step(fisher_and_grad, params, opt_state, key, model_samples, target_samples):

        grads = fisher_and_grad.grad_from_sample(params, model_samples, target_samples)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state


    def record_dists(model_log_probs):
        model_probs = jax.nn.softmax(model_log_probs)
        dist = l1_dist(target_probs, model_probs)
        div = kl_div(model_probs, target_probs)
        return dist, div

    for n_iter in range(TOTAL_ITER):
        # Main training loop
        model_log_probs = fisher_and_grad.log_probs_func(params)
        model_sampler = ExactSampler(model_log_probs)

        # Record
        if n_iter % RECORD_PER == 0:
            dist, div = record_dists(model_log_probs)

            l1_dists.append([n_iter, dist])
            kl_divs.append([n_iter, div])

            print(f'{n_iter}\tkl_div={div}\tl1_dist={dist}', flush=True)

        key, key_model, key_target = jrnd.split(key, 3)
        model_samples = jnp.array([to_bin_array(N, i) for i in model_sampler.sample(key_model, MINBATCH_SIZE)])
        target_samples = jnp.array([to_bin_array(N, i) for i in target_sampler.sample(key_target, MINBATCH_SIZE)])

        key, key_r = jrnd.split(key)

        params, opt_state = step(fisher_and_grad,params, opt_state, key_r, model_samples, target_samples)

    # Final record
    dist, div = record_dists(model_log_probs)
    l1_dists.append([TOTAL_ITER, dist])
    kl_divs.append([TOTAL_ITER, div])

    np.save('l1_dist.npy', l1_dists)
    np.save('kl_div.npy', kl_divs)
