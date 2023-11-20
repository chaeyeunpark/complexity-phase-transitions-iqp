import jax
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrnd
import jax.nn
from functools import partial
from tree_utils import reshape_tree_like, to_list
from numpy_bin_tools import to_bin_array

from typing import Iterable

import jax.tree_util as jtu

class ExactSampler:
    _log_probs: jnp.array

    def __init__(self, log_probs):
        self._log_probs = log_probs

    def sample(self, key, batch_size):
        indices = jax.random.categorical(key, self._log_probs, shape=(batch_size,))
        return indices

    def custom_sample(self, key, batch_size):
        accum = jnp.cumsum(jnp.exp(self._log_probs))
        rand_numbers = jrnd.uniform(key, (batch_size,), minval=0.0, maxval=accum[-1])
        indices = jnp.searchsorted(accum, rand_numbers)
        return indices

class SecondOrderMethod:
    _fisher: jnp.array
    _grad: jnp.array

    learning_rate: float
    beta1: float
    beta2: float

    t: int

    def __init__(self, *, learning_rate: float, beta1: float = 0.9, 
            beta2: float = 0.999):

        self._fisher = None
        self._grad = None

        self.t = 0
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    def update_momemtum(self, new_fisher, new_grad):
        if self._fisher is None:
            self._fisher = new_fisher
        else:
            self._fisher = self.beta2*self._fisher + (1-self.beta2)*new_fisher

        if self._grad is None:
            self._grad = new_grad
        else:
            self._grad = self.beta1*self._grad + (1-self.beta1)*new_grad

    def update(self, params):
        self.t += 1
        eps = 1e-3
        b = jnp.linalg.solve(self._fisher + eps*jnp.identity(self._fisher.shape[0]), self._grad)\
                * ((1. - jnp.power(self.beta2, self.t))/(1. - jnp.power(self.beta1, self.t)))
        b = reshape_tree_like(b, params)
        return jax.tree_util.tree_map(lambda x, y: x - self.learning_rate*y, params, b)

def flatten_SecondOrderMethod(som):
    flat_states = [som._fisher, som._grad]
    flat_hyperparams = [som.learning_rate, som.beta1, som.beta2]
    return ([flat_states, flat_hyperparams, som.t], None)

def unflatten_SecondOrderMethod(aux_data, flat_contents) -> SecondOrderMethod:
    flat_states, flat_hyperparams, t = flat_contents

    som = SecondOrderMethod(learning_rate = flat_hyperparams[0], beta1 = flat_hyperparams[1], beta2 = flat_hyperparams[2])

    som._fisher = flat_states[0]
    som._grad = flat_states[1]
    som.t = t
    return som

jax.tree_util.register_pytree_node(
    SecondOrderMethod, flatten_SecondOrderMethod, unflatten_SecondOrderMethod)

class FisherAndGrad:
    _N: int
    _model: nn.Module
    _full_confs: jnp.array

    def __init__(self, N, model):
        self._N = N
        self._model = model
        self._jitted_model = jax.jit(lambda p, x: model.apply(p, x))
        self._jitted_jac = jax.jit(lambda p, x: jax.jacrev(model.apply)(p, x))
        self._full_confs = jnp.array([to_bin_array(N, x) for x in range(1 << N)], dtype = jnp.int_)

    def log_probs_func(self, params):
        return self._jitted_model(params, self._full_confs).flatten()

    def from_full_confs(self, params, model_probs, target_probs):
        fullbatch_size = len(model_probs)
        MINBATCH_SIZE = 1024
        total_params = sum([p.size for p in jtu.tree_leaves(params)])

        assert fullbatch_size % MINBATCH_SIZE == 0
        assert len(model_probs) == len(target_probs)

        grad_pos = jnp.zeros((total_params,))
        grad_neg = jnp.zeros((total_params,))
        fisher = jnp.zeros((total_params, total_params))

        for i in range(fullbatch_size // MINBATCH_SIZE):
            batch_sl = slice(MINBATCH_SIZE*i, MINBATCH_SIZE*(i+1))
            jac = self._jitted_jac(params, self._full_confs[batch_sl])
            jac = jnp.hstack([g.reshape(MINBATCH_SIZE, -1) for g in jtu.tree_leaves(jac)])

            grad_pos += jnp.matmul(model_probs[batch_sl], jac)
            grad_neg += jnp.matmul(target_probs[batch_sl], jac)

            fisher += jnp.einsum('i,ij,ik->jk', model_probs[batch_sl], jac, jac)

        fisher -= jnp.outer(grad_pos, grad_pos)

        return fisher, grad_pos - grad_neg

    def from_sample(self, params, model_samples, target_samples):
        minbatch_size = len(model_samples)
        jac = self._jitted_jac(params, model_samples)
        jac = jnp.hstack([g.reshape(minbatch_size, -1) for g in jtu.tree_leaves(jac)])

        grad_pos = jnp.mean(jac, axis = 0)
        jac = jac - grad_pos[None, :]
        fisher = jnp.matmul(jac.T, jac) / minbatch_size

        f = lambda p: jnp.mean(self._jitted_model(p, target_samples))
        grad_neg = to_list(jax.grad(f)(params))

        return fisher, grad_pos - grad_neg

    def grad_from_sample(self, params, model_samples, target_samples):
        g = lambda p: jnp.mean(self._jitted_model(p, model_samples))
        grad_pos = jax.grad(g)(params)

        f = lambda p: jnp.mean(self._jitted_model(p, target_samples))
        grad_neg = jax.grad(f)(params)

        return jax.tree_util.tree_map(lambda x, y: x-y, grad_pos, grad_neg)
        #return grad_pos - grad_neg
