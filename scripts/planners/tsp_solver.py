from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from kristofedes import *
from P1_astar import *
from path_smoother import *

def tsp_solver(state_min, state_max, occupancy, plan_resolution, v_des, spline_alpha, traj_dt, locations):
    #print(locations)
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = compute_path_cost(state_min, state_max, occupancy, plan_resolution, v_des, spline_alpha, traj_dt, locations[i], locations[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    edges = kristofedes(distance_matrix)
    #print(edges)
    #G = nx.Graph(edges)
    #nx.draw(G, pos=locations)
    #plt.show()
    return edges, distance_matrix

def compute_path_cost(state_min, state_max, occupancy, plan_resolution, v_des, spline_alpha, traj_dt, loc1, loc2):
    problem = AStar(state_min, state_max, loc1,loc2, occupancy, plan_resolution)
    success =  problem.solve()
    if not success:
        print("TSP Planning failed")
        return euclidean(loc1, loc2)/v_des
    else:
        planned_path = problem.path
        traj_new, t_new = compute_smoothed_traj(planned_path, v_des, spline_alpha, traj_dt)
        return t_new[-1]
