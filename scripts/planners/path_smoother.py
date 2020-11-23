import numpy as np
import scipy.interpolate as itp

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # raise NotImplementedError # REPLACE THIS FUNCTION WITH YOUR IMPLEMENTATION
    path = np.array(path)
    x_old = path[:,0]
    y_old = path[:,1]
    n = np.shape(x_old)[0]
    # assert np.shape(y_old) == np.shape(x_old)
    t = np.zeros((n,))
    t_old = np.zeros((n,))
    # assert np.shape(x_old[1:] - x_old[:-1]) == (n-1,)
    t[1:] = np.linalg.norm(np.column_stack((x_old[1:] - x_old[:-1], y_old[1:]-y_old[:-1])), axis = 1) / V_des
    for i in range(n):
        t_old[i] = np.sum(t[:i+1])
    tck_x = itp.splrep(t_old, x_old, s = alpha)
    tck_y = itp.splrep(t_old, y_old, s = alpha)
    t_smoothed = np.arange(0.0, max(t_old), dt)
    n_new = np.shape(t_smoothed)[0]
    traj_smoothed = np.zeros((n_new,7))
    traj_smoothed[:, 0] = itp.splev(t_smoothed, tck_x)
    traj_smoothed[:, 1] = itp.splev(t_smoothed, tck_y)
    traj_smoothed[:, 3] = itp.splev(t_smoothed, tck_x, der = 1)
    traj_smoothed[:, 5] = itp.splev(t_smoothed, tck_x, der = 2)
    traj_smoothed[:, 4] = itp.splev(t_smoothed, tck_y, der = 1)
    traj_smoothed[:, 6] = itp.splev(t_smoothed, tck_y, der = 2)
    traj_smoothed[:, 2] = np.arctan2(traj_smoothed[:,4], traj_smoothed[:,3])
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
