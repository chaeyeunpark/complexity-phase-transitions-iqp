import numba
import numpy as np
import jax.numpy as jnp
import jax

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

def integrate(x0, x1, func, step = 100):
    xs = np.linspace(x0, x1, step)
    width = xs[1] - xs[0]
    ys = (width/6) * (func(xs[:-1]) + 4*func((xs[:-1] + xs[1:])/2) + func(xs[1:]))
    return sum(ys)


def calc_p(bins, p_func):
    p = []
    for i in range(len(bins)-1):
        p.append(integrate(bins[i], bins[i+1], p_func, 10))
    return np.array(p)

'''
:param p_k : observed distribution
'''
def kl_div(bins, p_k, q_k):
    p_k = jnp.clip(p_k, a_min = 1e-12, a_max = None)
    return jnp.dot(p_k, np.log(p_k/q_k))

@numba.jit(nopython=True)
def to_dist(bins, data):
    occur = np.zeros(len(bins)-1, dtype=np.int64)

    indices = np.searchsorted(bins, data)
    for i in indices:
        if i <= len(occur):
            occur[i-1] += 1

    return occur.astype(np.float64) / np.sum(occur)

"""
@jax.jit
def abs2(arr):
    return jnp.real(arr)**2 + jnp.imag(arr)**2
"""

def popcount(n):
    n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
    n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
    n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
    n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
    n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
    n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32) # This last & isn't strictly necessary.
    return n

