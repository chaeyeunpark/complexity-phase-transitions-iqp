import numpy as np
import numpy.random as nrd
import pennylane as qml
import math
import sys

def graph_edges(N, q):
    G = []
    for i in range(N-1):
        for j in range(i+1, N):
            G.append([i,j])
    selected = nrd.random(len(G)) < q

    return [G[idx] for idx, c in enumerate(selected) if c]

def create_circuit(N, edges):
    ini_st = np.ones(2**N, dtype=np.complex256) / np.sqrt(2**N)

    def circuit(phi, theta):
        qml.QubitStateVector(ini_st, wires = range(N))
        for i in range(N):
            qml.RZ(2*phi[i], wires=[i])

        for eidx, edge in enumerate(edges): 
            qml.IsingZZ(2*theta[eidx], wires=edge)
        return qml.state()

    return circuit

def test_edges():
    N = 20

    len_edges = []
    for i in range(1000):
        edges = graph_edges(N, 0.1)
        edges.sort()
        for i in range(len(edges)-1):
            assert edges[i] != edges[i+1]
        len_edges.append(len(edges))

    print(len_edges)
    print(0.1*(N*(N-1)//2))


def test():
    N = 14
    edges = graph_edges(N, 0.4)
    circuit = create_circuit(N, edges)

    dev1 = qml.device('lightning.qubit', wires = N)
    dev2 = qml.device('lightning.gpu', wires = N)

    qnode1 = qml.QNode(circuit, dev1, diff_method=None)
    qnode2 = qml.QNode(circuit, dev2, diff_method=None)

    rng = nrd.default_rng(1337)

    phi = 2*math.pi*rng.random(N)
    theta = 2*math.pi*rng.random(len(edges))

    st1 = qnode1(phi, theta)
    st2 = qnode2(phi, theta)

    assert np.allclose(st1, st2)
    print(np.abs(st1 - st2))

if __name__ == "__main__":
    #test_edges()
    test()
