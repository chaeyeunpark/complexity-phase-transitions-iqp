import pennylane as qml
from erdos_renyi_iqp import graph_edges, create_circuit
from numpy_bin_tools import walsh_hadamard
import sys
import numpy as np
import numpy.random as nrd
import math
import jax.numpy as jnp
import jax
from mpi4py import MPI
from utils import to_dist, abs2
from ham_weights_computer import HamWeightsComputer

import argparse

comm = MPI.COMM_WORLD

if __name__ == '__main__':
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    parser = argparse.ArgumentParser(
            description = 'Run random IQP circuit')

    parser.add_argument('N', type=int, metavar='N', help='Number of qubits')
    parser.add_argument('--q', required=False, type=float)
    parser.add_argument('--device', default='lightning.qubit', help="Which device to use to run the calculation")

    args = parser.parse_args()

    N = args.N
    nqs = [idx * 0.02 for idx in range(mpi_rank, 101, mpi_size)]
    qs = np.array(nqs) / N

    device = args.device

    dims = 2 ** N

    if N > 22:
        NUM_ITER = 2 ** 10
    else:
        NUM_ITER = 2 ** 12

    rng = nrd.default_rng(1337 + mpi_rank)
    dev = qml.device(device, wires = N)

    ham_weights_computer = HamWeightsComputer(N)

    for q in qs:
        print(f"Processing q={q} at rank={mpi_rank}", flush=True)

        prob_ham_weights = []

        for i in range(NUM_ITER):
            edges = graph_edges(N, q)
            circuit = qml.QNode(create_circuit(N, edges), dev, diff_method=None)
            phi = 2*math.pi * rng.random(N)
            theta = 2*math.pi * rng.random(len(edges))
            st = circuit(phi, theta)
            st = jnp.array(walsh_hadamard(st))

            prob = abs2(st)
            prob_ham_weights.append(ham_weights_computer(prob))

        jnp.save(f'IQP_ZZ_N{N}_NQ{int(1000*N*q):04d}_HAM_WEIGHTS.npy', prob_ham_weights)
