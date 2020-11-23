import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from blossom import *

INF = 10**9


def prima(W):
    n = len(W)
    g = [[] for _ in range(n)]
    used = [False for _ in range(n)]
    min_e = [INF for _ in range(n)]
    pred = [-1 for _ in range(n)]
    min_e[0] = 0
    for _ in range(n):
        v = min((i for i in range(n) if not used[i]), key=lambda i: min_e[i])
        used[v] = True
        if pred[v] != -1:
            g[v].append(pred[v])
            g[pred[v]].append(v)
        for to in range(n):
            if W[v][to] < min_e[to]:
                min_e[to] = W[v][to]
                pred[to] = v
    return g


def find_euler(g):
    n = len(g)
    deg = [sum(row) for row in g]
    assert all(d % 2 == 0 for d in deg)
    first = 0
    while not deg[first]:
        first += 1
    st = [first]
    res = []
    while st:
        v = st[-1]
        try:
            i = 0
            while not g[v][i]: i += 1
            g[v][i] -= 1
            g[i][v] -= 1
            st.append(i)
        except IndexError:
            res.append(v)
            st.pop()

    return res

def kristofedes(a):
    n = len(a)
    g = prima(a)
    odd_vertices = [i for i in range(n) if len(g[i]) % 2]
    p = blossom(odd_vertices, a)
    new_g = [[0] * n for i in range(n)]
    for i, j in p:
        new_g[i][j] += 1
        new_g[j][i] += 1
    for i in range(n):
        for j in g[i]:
            new_g[i][j] += 1
            #new_g[j][i] += 1
    path = find_euler(new_g)
    new_path = []
    path.pop()
    for v in path:
        if v not in new_path:
            new_path.append(v)
    path = new_path
    path.append(path[0])
    edges = [(path[i-1], path[i]) for i in range(len(path))]

    return edges
