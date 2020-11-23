import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import plot_line_segments

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        #check_lo, check_hi = [False, False]
        #check_lo = x[0] >= self.statespace_lo[0] and x[1] >= self.statespace_lo[1]
        #check_hi = x[0] <= self.statespace_hi[0] and x[1] <= self.statespace_hi[1]
        #return check_lo and check_hi and self.occupancy.is_free(x)
        if (x[0] < self.statespace_lo[0] or x[0] > self.statespace_hi[0]
            or x[1] < self.statespace_lo[1] or x[1] > self.statespace_hi[1]):
            return False
        if (self.occupancy.is_free(x)):
            return True
        return False
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line.
        """
        ########## Code starts here ##########
        return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by simply adding/subtracting self.resolution
               from x, numerical error could creep in over the course of many
               additions and cause grid point equality checks to fail. To remedy
               this, you should make sure that every neighbor is snapped to the
               grid as it is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        #res = self.resolution
        # horizontal:
        #neighbors_temp = np.array([x[0], x[1] + res])
        #neighbors_temp = np.vstack((neighbors_temp, np.array([x[0], x[1] - res])))
        # vertical:
        #neighbors_temp = np.vstack((neighbors_temp, [x[0] + res, x[1]]))
        #neighbors_temp = np.vstack((neighbors_temp, [x[0] - res, x[1]]))
        # diagonal:
        #neighbors_temp = np.vstack((neighbors_temp, [x[0] + res, x[1] + res]))
        #neighbors_temp = np.vstack((neighbors_temp, [x[0] - res, x[1] - res]))
        #neighbors_temp = np.vstack((neighbors_temp, [x[0] + res, x[1] - res]))
        #neighbors_temp = np.vstack((neighbors_temp, [x[0] - res, x[1] + res]))
        #for i in range(len(neighbors_temp)):
        #    neighbors_temp[i] = self.snap_to_grid(neighbors_temp[i])
        #    if self.is_free(neighbors_temp[i]):
        #        neighbors.append((neighbors_temp[i][0], neighbors_temp[i][1]))
        up = self.snap_to_grid((x[0], x[1] + self.resolution))
        down = self.snap_to_grid((x[0], x[1] - self.resolution))
        left = self.snap_to_grid((x[0] - self.resolution, x[1]))
        right = self.snap_to_grid((x[0] + self.resolution, x[1]))
        left_up = self.snap_to_grid((x[0] - self.resolution, x[1] + self.resolution))
        left_down = self.snap_to_grid((x[0] - self.resolution, x[1] - self.resolution))
        right_up = self.snap_to_grid((x[0] + self.resolution, x[1] + self.resolution))
        right_down = self.snap_to_grid((x[0] + self.resolution, x[1] - self.resolution))
        grids = [up, down, left, right, left_up, left_down, right_up, right_down]
        for grid in grids:
            if (self.is_free(grid)):
                neighbors.append(grid)
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.array(self.path) * self.resolution
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0]*self.resolution, self.x_goal[0]*self.resolution], [self.x_init[1]*self.resolution, self.x_goal[1]*self.resolution], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", np.array(self.x_init)*self.resolution + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal)*self.resolution + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        #while len(self.open_set) > 0:
        #    x_c = self.find_best_est_cost_through()
        #    if x_c == self.x_goal:
        #        self.path = self.reconstruct_path()
        #        return True
        #    self.open_set.remove(x_c)
        #    self.closed_set.add(x_c)
        #    for x_n in self.get_neighbors(x_c):
        #        if x_n in self.closed_set:
        #            continue
        #        tent_cost_to_arrive = self.cost_to_arrive[x_c] + self.distance(x_c, x_n)
        #        if x_n not in self.open_set:
        #            self.open_set.add(x_n)
        #        elif tent_cost_to_arrive > self.cost_to_arrive[x_n]:
        #            continue
        #        self.came_from[x_n] = x_c
        #        self.cost_to_arrive[x_n] = tent_cost_to_arrive
        #        self.est_cost_through[x_n] = tent_cost_to_arrive + self.distance(x_n, self.x_goal)
        #return False
        count = 0
        while len(self.open_set) > 0:
            count += 1
            if count > 300:
                return False
            x_current = self.find_best_est_cost_through()
            # print(x_current)
            if (np.abs(np.array(x_current) - np.array(self.x_goal)).max() < 1e-5):
            #if (x_current[0] == self.x_goal[0] and x_current[1] == self.x_goal[1]):
                # no need to path reconstruction
                self.path = self.reconstruct_path()
                return True

            self.open_set.remove(x_current)
            self.closed_set.add(x_current)

            x_current_neighbors = self.get_neighbors(x_current)
            # print(x_current_neighbors)
            for x_neighbor in x_current_neighbors:
                if x_neighbor in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neighbor)
                if x_neighbor not in self.open_set:
                    self.open_set.add(x_neighbor)
                    self.cost_to_arrive[x_neighbor] = np.Inf
                    # require modifying cost to cost_to_arrive
                elif (tentative_cost_to_arrive > self.cost_to_arrive[x_neighbor]):
                    continue

                self.came_from[x_neighbor] = x_current
                self.cost_to_arrive[x_neighbor] = tentative_cost_to_arrive
                self.est_cost_through[x_neighbor] = tentative_cost_to_arrive + self.distance(x_neighbor, self.x_goal)

        return False

        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles"""
        for obs in self.obstacles:
            inside = True
            for dim in range(len(x)):
                if x[dim] < obs[0][dim] or x[dim] > obs[1][dim]:
                    inside = False
                    break
            if inside:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        for obs in self.obstacles:
            ax = fig.add_subplot(111, aspect='equal')
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))

