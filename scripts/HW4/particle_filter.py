import numpy as np
import scipy.linalg  # You may find scipy.linalg.block_diag useful
import scipy.stats  # You may find scipy.stats.multivariate_normal.pdf useful
import turtlebot_model as tb

EPSILON_OMEGA = 1e-3

class ParticleFilter(object):
    """
    Base class for Monte Carlo localization and FastSLAM.

    Usage:
        pf = ParticleFilter(x0, R)
        while True:
            pf.transition_update(u, dt)
            pf.measurement_update(z, Q)
            localized_state = pf.x
    """

    def __init__(self, x0, R):
        """
        ParticleFilter constructor.

        Inputs:
            x0: np.array[M,3] - initial particle states.
             R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.M = x0.shape[0]  # Number of particles
        self.xs = x0  # Particle set [M x 3]
        self.ws = np.repeat(1. / self.M, self.M)  # Particle weights (initialize to uniform) [M]
        self.R = R  # Control noise covariance (corresponding to dt = 1 second) [2 x 2]

    @property
    def x(self):
        """
        Returns the particle with the maximum weight for visualization.

        Output:
            x: np.array[3,] - particle with the maximum weight.
        """
        idx = self.ws == self.ws.max()
        x = np.zeros(self.xs.shape[1:])
        x[:2] = self.xs[idx,:2].mean(axis=0)
        th = self.xs[idx,2]
        x[2] = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        return x

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating self.xs.

        Inputs:
            u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.xs) should be updated.
        """
        ########## Code starts here ##########
        # TODO: Update self.xs.
        # Hint: Call self.transition_model().
        # Hint: You may find np.random.multivariate_normal useful.
        us = np.random.multivariate_normal(u, self.R, self.M)
	self.xs = self.transition_model(us, dt)
	

        ########## Code ends here ##########

    def transition_model(self, us, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        raise NotImplementedError("transition_model must be overridden by a subclass of EKF")

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        raise NotImplementedError("measurement_update must be overridden by a subclass of EKF")

    def resample(self, xs, ws):
        """
        Resamples the particles according to the updated particle weights.

        Inputs:
            xs: np.array[M,3] - matrix of particle states.
            ws: np.array[M,]  - particle weights.

        Output:
            None - internal belief state (self.xs, self.ws) should be updated.
        """
        r = np.random.rand() / self.M

        ########## Code starts here ##########
        # TODO: Update self.xs, self.ws.
        # Note: Assign the weights in self.ws to the corresponding weights in ws
        #       when resampling xs instead of resetting them to a uniform
        #       distribution. This allows us to keep track of the most likely
        #       particle and use it to visualize the robot's pose with self.x.
        # Hint: To maximize speed, try to implement the resampling algorithm
        #       without for loops. You may find np.linspace(), np.cumsum(), and
        #       np.searchsorted() useful. This results in a ~10x speedup.
	M = self.M
	u = np.sum(ws) * (r + np.linspace(0, 1, M, endpoint=False))
	c = np.cumsum(ws)
	index = np.searchsorted(c, u)
        self.xs = xs[index]
        self.ws = ws[index]
        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction).

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2I,]   - joint measurement mean.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        raise NotImplementedError("measurement_model must be overridden by a subclass of EKF")


class MonteCarloLocalization(ParticleFilter):

    def __init__(self, x0, R, map_lines, tf_base_to_camera, g):
        """
        MonteCarloLocalization constructor.

        Inputs:
                       x0: np.array[M,3] - initial particle states.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[2,J] - J map lines in columns representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as columns
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, R)

    def transition_model(self, us, dt):
        """
        Unicycle model dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """

        ########## Code starts here ##########
        # TODO: Compute g.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: A simple solution can be using a for loop for each partical
        #       and a call to tb.compute_dynamics
        # Hint: To maximize speed, try to compute the dynamics without looping
        #       over the particles. If you do this, you should implement
        #       vectorized versions of the dynamics computations directly here
        #       (instead of modifying turtlebot_model). This results in a
        #       ~10x speedup.
        # Hint: This faster/better solution does not use loop and does 
        #       not call tb.compute_dynamics. You need to compute the idxs
        #       where abs(om) > EPSILON_OMEGA and the other idxs, then do separate 
        #       updates for them
        x, y, theta = np.transpose(self.xs)
	V, omega = np.transpose(us)

        mask = np.abs(omega) > EPSILON_OMEGA

        new_theta = theta + omega*dt
        
	new_x = np.zeros(x.shape)
	new_y = np.zeros(y.shape)

        new_x[mask] = x[mask] + V[mask] * (np.sin(new_theta[mask]) - np.sin(theta[mask])) / omega[mask]
        new_y[mask] = y[mask] + V[mask] * (-np.cos(new_theta[mask]) + np.cos(theta[mask])) / omega[mask]

        mask = np.abs(omega) <= EPSILON_OMEGA
        new_x[mask] = x[mask] + V[mask] * (np.cos(theta[mask]) + np.cos(new_theta[mask])) / 2.0 * dt
        new_y[mask] = y[mask] + V[mask] * (np.sin(new_theta[mask]) + np.sin(theta[mask])) / 2.0 * dt

	g = np.vstack((new_x, new_y, new_theta)).transpose()
        ########## Code ends here ##########

        return g

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        xs = np.copy(self.xs)
        ws = np.zeros_like(self.ws)
        ########## Code starts here ##########
        # TODO: Compute new particles (xs, ws) with updated measurement weights.
        # Hint: To maximize speed, implement this without looping over the
        #       particles. You may find scipy.stats.multivariate_normal.pdf()
        #       useful.
        # Hint: You'll need to call self.measurement_model()
        vs, Q = self.measurement_model(z_raw, Q_raw)
	ws = scipy.stats.multivariate_normal.pdf(vs, cov = Q)
        ########## Code ends here ##########

        self.resample(xs, ws)

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature for each particle.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[M,2I]  - joint measurement mean for M particles.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        vs = self.compute_innovations(z_raw, np.array(Q_raw)) # [M, 2I]

        ########## Code starts here ##########
        # TODO: Compute Q.
        # Hint: You might find scipy.linalg.block_diag() useful
	Q = scipy.linalg.block_diag(*Q_raw)

        ########## Code ends here ##########

        return vs, Q

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: np.array[I,2,2] - I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            vs: np.array[M,2I] - M innovation vectors of size 2I
                                 (predicted map measurement - scanner measurement).
        """
        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        ########## Code starts here ##########
        # TODO: Compute vs (with shape [M x I x 2]).
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       observed line, find the most likely map entry (the entry with 
        #       least Mahalanobis distance).
        # Hint: To maximize speed, try to eliminate all for loops, or at least
        #       for loops over J. It is possible to solve multiple systems with
        #       np.linalg.solve() and swap arbitrary axes with np.transpose().
        #       Eliminating loops over J results in a ~10x speedup.
        #       Eliminating loops over I results in a ~2x speedup.
        #       Eliminating loops over M results in a ~5x speedup.
        #       Overall, that's 100x!
        # Hint: For the faster solution, you might find np.expand_dims(), 
        #       np.linalg.solve(), np.meshgrid() useful.

	# Dimensions
        I = np.shape(z_raw)[1]
        M = self.M       
        hs = self.compute_predicted_measurements() #[M, 2, J]
        J = np.shape(hs)[2]

	# Decomposition
	z_alpha = z_raw[0,:].reshape(1,I,1) # [1, I, 1]
	z_r = z_raw[1,:].reshape(1,I,1) # [1, I, 1]
	hs_alpha = hs[:,0,:].reshape(M,1,J) # [M, 1, J]
	hs_r = hs[:, 1, :].reshape(M,1,J) # [M, 1, J]
	z_alpha_mij = np.tile(z_alpha, (M,1,J))
	z_r_mij = np.tile(z_r, (M,1,J))
	hs_alpha_mij = np.tile(hs_alpha, (1,I,1))
	hs_r_mij = np.tile(hs_r, (1,I,1))

        Q_raw = np.linalg.inv(Q_raw)
        Q_mij_raw = Q_raw.reshape(1,I,1,2,2)
        Q_mij = np.tile(Q_mij_raw, (M,1,J,1,1))
	Q_mij_flattened = np.reshape(Q_mij, (M*I*J, 2, 2))	
        Q_diag = scipy.linalg.block_diag(*list(Q_mij_flattened))

	# Calculation
	v_alpha_mij = angle_diff(z_alpha_mij, hs_alpha_mij)
        v_r_mij = -hs_r_mij + z_r_mij
       	v_mij2 = np.stack((v_alpha_mij, v_r_mij),axis=3) # [M, I, J, 2]
	v_mij2_flattened = np.reshape(v_mij2, (M*I*J*2, 1)) # [2MIJ, 1]

	v_mij = np.reshape(v_mij2, (M*I*J, 2))
	v_trans_mij_diag = scipy.linalg.block_diag(*list(v_mij)) # [MIJ, 2MIJ]
        d_mij_flattened = np.matmul(v_trans_mij_diag, np.matmul(Q_diag, v_mij2_flattened))

        '''
	Q_mij_raw = Q_raw.reshape(1,I,1,2,2)
        Q_mij = np.tile(Q_mij_raw, (M,1,J,1,1))
	Q_mij_flattened = np.reshape(Q_mij, (M*I*J, 2, 2))	
        Q_diag = scipy.linalg.block_diag(*list(Q_mij_flattened))

	# Calculation
	v_alpha_mij = angle_diff(z_alpha_mij, hs_alpha_mij)
        v_r_mij = -hs_r_mij + z_r_mij
       	v_mij2 = np.stack((v_alpha_mij, v_r_mij),axis=3) # [M, I, J, 2]
	v_mij2_flattened = np.reshape(v_mij2, (M*I*J*2, 1)) # [2MIJ, 1]

	v_mij = np.reshape(v_mij2, (M*I*J, 2))
	v_trans_mij_diag = scipy.linalg.block_diag(*list(v_mij)) # [MIJ, 2MIJ]

	d_mij_flattened = np.matmul(v_trans_mij_diag, np.linalg.solve(Q_diag, v_mij2_flattened))
        '''
	d_mij = np.reshape(d_mij_flattened, (M,I,J))
	d_min_index = np.argmin(d_mij, axis=-1) # [M, I]
	d_min_index_mij2 = d_min_index.reshape(M,I,1,1)
        vs = np.take_along_axis(v_mij2, d_min_index_mij2, axis=2).reshape(M,I,2) # [M, I, 2]
        ########## Code ends here ##########

        # Reshape [M x I x 2] array to [M x 2I]
        return vs.reshape((self.M,-1))  # [M x 2I]

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Output:
            hs: np.array[M,2,J] - J line parameters in the scanner (camera) frame for M particles.
        """
        ########## Code starts here ##########
        # TODO: Compute hs.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       map line, transform to scanner frmae using tb.transform_line_to_scanner_frame()
        #       and tb.normalize_line_parameters()
        # Hint: To maximize speed, try to compute the predicted measurements
        #       without looping over the map lines. You can implement vectorized
        #       versions of turtlebot_model functions directly here. This
        #       results in a ~10x speedup.
        # Hint: For the faster solution, it does not call tb.transform_line_to_scanner_frame()
        #       or tb.normalize_line_parameters(), but reimplement these steps vectorized.
        
	alpha, r = self.map_lines # (J,)
	J = np.shape(alpha)[0]
    	x_val, y_val, theta_val = np.transpose(self.xs) # (M,)
    	x_base, y_base, theta_base = self.tf_base_to_camera # fixed
	x_cam = np.cos(theta_val)*x_base - np.sin(theta_val)*y_base + x_val # (M,)
	y_cam = np.sin(theta_val)*x_base + np.cos(theta_val)*y_base + y_val # (M,)

	alpha_MJ = np.tile(alpha,(self.M,1))
        theta_val_MJ = np.tile(theta_val.reshape((self.M,1)), (1,J))
        alpha_in_cam = alpha_MJ - theta_val_MJ - theta_base # (M, J)

	r_MJ = np.tile(r,(self.M, 1))
	r_in_cam = r - np.matmul(x_cam.reshape((self.M, 1)), np.cos(alpha).reshape((1, J))) - np.matmul(y_cam.reshape((self.M, 1)), np.sin(alpha).reshape((1, J))) # (M, J)
	
        mask = r_in_cam < 0
        r_in_cam[mask] *= -1
        alpha_in_cam[mask] += np.pi
        alpha_in_cam = (alpha_in_cam + np.pi) % (2 * np.pi) - np.pi

        hs = np.stack((alpha_in_cam, r_in_cam), axis = 1)
	
        ########## Code ends here ##########

        return hs

