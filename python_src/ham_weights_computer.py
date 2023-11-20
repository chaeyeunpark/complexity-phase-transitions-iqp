from utils import popcount
from numpy_bin_tools import walsh_hadamard
import numpy as np
import jax.numpy as jnp
import math

class HamWeightsComputer:
    def __init__(self, N: int):
        confs = list(range(0, 2 ** N))
        self.N = N

        m = [[] for _ in range(N+1)]
        for c in confs:
            m[popcount(c)].append(c)

        for i in range(N+1):
            m[i] = jnp.array(m[i])
        self.confs_with_hamming_weights = m

    def __call__(self, prob):
        N = self.N
        assert len(prob) == 2**N
        res = np.zeros((N+1,), dtype=np.float64)
        Js = walsh_hadamard(jnp.log(prob)) / math.pow(2, N/2)
        Js = jnp.abs(Js)

        for i in range(N+1):
            res[i] += jnp.sum(Js[self.confs_with_hamming_weights[i]])

        return jnp.array(res)

