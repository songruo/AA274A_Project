import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

from kristofedes import *

n = 10
pos = np.random.random((n, 2))
print(pos)
a = [[euclidean(u, v) for v in pos] for u in pos]

edges = kristofedes(a)
print(edges)
G = nx.Graph(edges)
nx.draw(G, pos=pos)
plt.show()
