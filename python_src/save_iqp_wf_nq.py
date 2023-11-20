import pennylane as qml
from erdos_renyi_iqp import graph_edges, create_circuit
from numpy_bin_tools import walsh_hadamard
import sys
import numpy as np
import numpy.random as nrd
import math

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run random IQP circuit')

    parser.add_argument('N', type=int, metavar='N', help='Number of qubits')
    parser.add_argument('--nq', required=True, type=float)
    parser.add_argument('--device', default='lightning.qubit', help="Which device to use to run the calculation")

    args = parser.parse_args()
    N = args.N
    nq = args.nq
    device = args.device

    rng = nrd.default_rng(1337)
    dev = qml.device(device, wires = N)

    states = []

    q = nq / N

    for i in range(32):
        edges = graph_edges(N, q)
        circuit = qml.QNode(create_circuit(N, edges), dev, diff_method=None)
        phi = 2*math.pi * rng.random(N)
        theta = 2*math.pi * rng.random(len(edges))
        st = circuit(phi, theta)
        st = np.array(walsh_hadamard(st))
        states.append(st)

    np.save('IQP_ZZ_N{}_NQ{:04d}.npy'.format(N, int(nq*100)), states)
