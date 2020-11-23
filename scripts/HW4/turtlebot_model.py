import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    x, y, theta = xvec
    V, omega = u
    
    new_theta = theta + omega*dt
    
    if abs(omega) < EPSILON_OMEGA:
        new_x = x + V*(np.cos(theta)+np.cos(new_theta))/2*dt
        new_y = y + V*(np.sin(theta)+np.sin(new_theta))/2*dt
        Gx = np.array([[1, 0, V*(-np.sin(theta) - np.sin(new_theta))/2*dt],
                       [0, 1, V*(np.cos(theta) + np.cos(new_theta))/2*dt],
                       [0, 0, 1]])
        Gu = np.array([[(np.cos(theta)+np.cos(new_theta))/2*dt, -V*np.sin(new_theta)/2*dt*dt],
                       [(np.sin(theta)+np.sin(new_theta))/2*dt, V*np.cos(new_theta)/2*dt*dt],
                       [0, dt]])
    else:
        new_x = x + V*(np.sin(new_theta) - np.sin(theta))/omega
        new_y = y + V*(-np.cos(new_theta) + np.cos(theta))/omega
        Gx = np.array([[1, 0, V*(np.cos(new_theta) - np.cos(theta))/omega],
                       [0, 1, V*(np.sin(new_theta) - np.sin(theta))/omega],
                       [0, 0, 1]])
        Gu = np.array([[(np.sin(new_theta) - np.sin(theta))/omega, V/(omega**2)*(omega*np.cos(new_theta)*dt - np.sin(new_theta) + np.sin(theta))],
                       [(-np.cos(new_theta) + np.cos(theta))/omega, V/(omega**2)*(omega*np.sin(new_theta)*dt + np.cos(new_theta) - np.cos(theta))],
                       [0, dt]])
    
    g = np.array([new_x, new_y, new_theta])

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    x_val, y_val, theta_val = x
    x_base, y_base, theta_base = tf_base_to_camera

    #R_world2base = np.array([[np.cos(theta_val), np.sin(theta_val)],
    #                         [-np.sin(theta_val), np.cos(theta_val)]])
    #[x_val,y_val] = -np.matmul(R_world2base, np.array([x_val, y_val]))

    #R_base2cam = np.array([[np.cos(theta_base), np.sin(theta_base)],
    #                         [-np.sin(theta_base), np.cos(theta_base)]])
    #[x_base,y_base] = -np.matmul(R_base2cam, np.array([x_base, y_base]))

    # T_worldToBase = np.array([[np.cos(theta_val), -np.sin(theta_val), x_val], 
    #			      [np.sin(theta_val), np.cos(theta_val), y_val],
    #			      [0.,0.,1.]])
    # T_baseToCamera = np.linalg.inv(np.array([[np.cos(theta_base), -np.sin(theta_base), x_base],
    #			                     [np.sin(theta_base), np.cos(theta_base), y_base],
    #                 	                     [0.,0.,1.]]))
    # line_cart = np.array([r*np.cos(alpha), r*np.sin(alpha), 1.])

    # camara_base = np.array([x_base, y_base, 1])
    # x_world = np.matmul(T_worldToBase, camara_base)

    # line_camera = np.dot(np.matmul(T_baseToCamera, T_worldToBase), line_cart)
    alpha_in_cam = alpha  - theta_val - theta_base
    # x_cam, y_cam, theta_cam in world frame
    #[x_cam, y_cam] = np.matmul(R_world2base.transpose(), np.array([x_base, y_base])) + np.array([x_val, y_val])
    x_cam = np.cos(theta_val)*x_base - np.sin(theta_val)*y_base + x_val
    y_cam = np.sin(theta_val)*x_base + np.cos(theta_val)*y_base + y_val
    # r_in_cam = r - np.sqrt(x_cam**2+y_cam**2)*np.cos(alpha-np.arctan2(y_cam,x_cam))
    r_in_cam = r - np.cos(alpha) * x_cam - y_cam * np.sin(alpha)
    h = np.array([alpha_in_cam, r_in_cam])

    Hx = np.zeros((2,3))
    
    Hx[0,0] = 0.
    Hx[0,1] = 0.
    Hx[0,2] = -1.

    d_cam = np.sqrt(x_cam**2 + y_cam**2)
    theta_cam = np.arctan2(y_cam, x_cam)

    Hx[1, 0] = -x_cam/d_cam*np.cos(alpha - theta_cam) + d_cam * \
        np.sin(alpha - theta_cam)*(y_cam/(x_cam*x_cam + y_cam*y_cam))
    Hx[1, 1] = -y_cam/d_cam*np.cos(alpha - theta_cam) - d_cam * \
        np.sin(alpha - theta_cam)*(x_cam/(x_cam*x_cam + y_cam*y_cam))
    Hx[1, 2] = Hx[1, 0]*(-np.sin(theta_val)*x_base - np.cos(theta_val)*y_base) + \
        Hx[1, 1]*(np.cos(theta_val)*x_base - np.sin(theta_val)*y_base)
    #Hx[1,0] = -np.cos(alpha)*np.cos(theta_val) - np.sin(alpha)*np.sin(theta_val)
    #Hx[1,1] = np.cos(alpha)*np.sin(theta_val) - np.sin(alpha)*np.cos(theta_val)
    #Hx[1,2] = y_base + y_val*np.cos(theta_base) + 2*(x_val**2+y_val**2)*np.cos(2*theta_val+theta_base) + x_val*np.sin(theta_base)
    
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
