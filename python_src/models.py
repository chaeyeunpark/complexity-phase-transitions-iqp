import jax
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrnd
import itertools

class RBM(nn.Module):
    n_hidden: int
    dtype: jnp.dtype
    initializer: str = None

    def setup(self):
        if self.initializer is None:
            initializer = jax.nn.initializers.lecun_normal()
        else:
            initializer = getattr(jax.nn.initializers, self.initializer)()

        self.fc1 = nn.Dense(self.n_hidden, use_bias = True, dtype=self.dtype, 
                kernel_init = initializer)
        self.fc2 = nn.Dense(1, use_bias = False, dtype=self.dtype, kernel_init = initializer)

    def __call__(self, x):
        h = self.fc1(x)
        h =  jnp.sum(h - nn.log_sigmoid(h), axis = -1, keepdims = True)
        x = self.fc2(x)
        return x + h

class FullyConnected(nn.Module):
    n_hidden: int
    dtype: jnp.dtype
    initializer: str = None

    def setup(self):
        if self.initializer is None:
            initializer = jax.nn.initializers.lecun_normal(dtype = self.dtype)
        else:
            initializer = getattr(jax.nn.initializers, self.initializer)(dtype = self.dtype)

        self.fc1 = nn.Dense(self.n_hidden, use_bias = True, dtype=self.dtype, kernel_init = initializer)
        self.fc2 = nn.Dense(1, use_bias = False, dtype=self.dtype, kernel_init = initializer)

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        return x

def n_choose_k(n: int, k: int):
    m = 1
    for i in range(n-k+1, n+1):
        m *= i
    for i in range(1,k+1):
        m //= i
    return m

class LocalIsing(nn.Module):
    N: int
    max_k: int = 2
    dtype: jnp.dtype = jnp.float64

    def setup(self):
        pos = list(range(self.N))
        combs = []
        for k in range(2, self.max_k+1):
            combs.append(jnp.array(list(itertools.combinations(pos, k))))

        self.combs = combs

    @nn.compact
    def __call__(self, x):
        N = x.shape[-1]

        energy = jnp.zeros((x.shape[0],), dtype = self.dtype)

        J1 = self.param(f"J_1", lambda rng: 1e-4*jrnd.normal(rng, (N, )))
        energy += x @ J1

        for k in range(2, self.max_k+1):
            len_combs = n_choose_k(N, k)
            Jk = self.param(f'J_{k}', lambda rng, shape: 1e-4*jrnd.normal(rng, shape, dtype=self.dtype), (len_combs,))

            z_k = []

            for c in self.combs[k-2]:
                z_k.append(jnp.prod(x[:,c], axis=1))
            
            z_k = jnp.array(z_k)
            energy += Jk @ z_k

        return energy
