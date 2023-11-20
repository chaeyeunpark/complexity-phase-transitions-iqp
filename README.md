# Complexity phase transitions in instantaneous quantum polynomial-time circuits
Source code for "complexity phase transitions in instantaneous quantum polynomial-time circuits." 


## Reproducing data


### Obtaining data for Fig. 1

We used our Kokkos-based C++ code (see `cpp_src` directory) to generate data for Fig. 1 in the main text. To compile the code, you need a C++ compiler with C++17 support (e.g., GCC >= 10.0 must work). A GPU backend should be used to obtain results for more than 30 qubits. For example, one can compile our code for CUDA as follows:
```
$ mkdir build_cuda && cd build_cuda
& cmake -DKokkos_ENABLE_CUDA=ON ..
```

The compiled executable `iqp_dist_scaled_mpi` is MPI-enabled and runnable on a cluster.


### Obtaining data for Fig. 2(a)


For Fig. 2(a), a Python script is used. In addition to packages in `requirements.txt`, you need to install `jax` (see [Jax installation guide](https://jax.readthedocs.io/en/latest/installation.html)) and `mpi4py` (see [MPI for python installation page](https://mpi4py.readthedocs.io/en/stable/install.html#)). Then running `python_src/iqp_ham_weights_nq.py` will generate results. For example,
```
$ python3 python_src/iqp_ham_weights_nq.py 20
```
will produce results for 20 qubits.

### Obtaining data for Fig. 2(b)

For Fig. 2(b), you first need to generate a quantum state for different $N$ and $qN$. We provide `python_src/save_iqp_wf_nq.py` for this purpose. Running this script with arguments $N$ and $qN$ will save the data to a file. For example,
```
$ python3 python_src/save_iqp_wf_nq.py 16 0.25
```
will create a file `IQP_ZZ_N16_NQ025.npy` which contains wavefunctions for $N=16$ and $qN=0.25$ for $32$ circuit instances. Then we can train a neural network using `python_src/learn_iqp_ngd.py`:
```
$ python3 python_src/learn_iqp_ngd.py --data-path IQP_ZZ_N16_NQ025.npy --learning-rate 0.0005 --beta2 0.999 --gamma 10 --idx 0
```
Here, the argument `--idx` indicates that the wavefunction in the data file of the given index will be used as train data. Thus, in our case, it is an integer between 0 to 31. Arguments `--learning-rate` and `--beta2` controls the hyperparameters, and `--gamma` determines the size of the network to use (see the paper).
