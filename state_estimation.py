import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import embed

class Robot:
    def __init__(self, radius, mu0, Sigma0, proc_noise_std, obs_noise_std):
        # Just for the looks
        self.radius = radius

        #  Defining parameters for KF algorithm
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        self.proc_noise_std = proc_noise_std
        self.obs_noise_std = obs_noise_std

        # Process noise covariance (R)
        self.R = np.diag(self.proc_noise_std ** 2)  # process noise covariance

        # Observation model (C)
        self.C = np.eye(3)

        # Standard deviations for the noise in x, y, and theta (observation or sensor model noise)
        # Observation noise covariance (Q)
        self.Q = np.diag(self.obs_noise_std ** 2)
        
    def action_model(self, At, Bt, ut, xt_1):
        # xt = At*xt-1 + Bt*ut + wt
        # wt is gaussian noise with covariance R (process noise/ actuators noise)
        wt = np.random.multivariate_normal(np.zeros(3), self.R)
        xt = At @ xt_1 + Bt @ ut + wt
        return xt

    def sensor_model(self, Ct, xt):
        # zt = Ct*xt + vt
        # vt is gaussian noise with covariance Q (sensor noise)
        vt = np.random.multivariate_normal(np.zeros(3), self.Q)
        zt = Ct @ xt + vt
        return zt

    def dynamic_model(self, theta):
        # Input: mu_t-1
        # At = np.eye(3)
        # Bt = np.array([[np.cos(self.mu[2]), 0], [np.sin(self.mu[2]), 0], [0, 1]])
        At = np.eye(3)
        Bt = np.array([
            [np.cos(theta), 0], 
            [np.sin(theta), 0], 
            [0, 1]
        ])
        return At, Bt
    
    def g_nonlin_motion(self, mu, u):
        # mu, u are np vectors of size 3x1 and 2x1 respectively
        x, y, theta = mu
        v, w = u
        x_new = x + v*np.cos(theta)
        y_new = y + v*np.sin(theta)
        theta_new = theta + w
        return np.array([x_new, y_new, theta_new])
    
    def G_nonlin_motion(self, mu, u):
        # mu, u are np vectors of size 3x1 and 2x1 respectively
        x, y, theta = mu
        v, w = u
        G = np.array([
            [1, 0, -v*np.sin(theta)],
            [0, 1, v*np.cos(theta)],
            [0, 0, 1]
        ])
        return G
    
    def h_observation(self, mu):
        # mu is np vector of size 3x1
        return mu
    
    def H_observation(self):
        return np.eye(3)

    def kalman_filter(self, mu, cov, u_t, z_t, At, Bt):
        # Prediction step
        mu_t_bar = At @ mu + Bt @ u_t
        sigma_t_bar = At @ cov @ At.T + self.R

        # Kalman Gain calculation
        K = sigma_t_bar @ self.C.T @ np.linalg.inv(self.C @ sigma_t_bar @ self.C.T + self.Q)

        # Update step
        mu = mu_t_bar + K @ (z_t - self.C @ mu_t_bar)
        sigma = (np.eye(3) - K @ self.C) @ sigma_t_bar

        return mu, sigma
    
    def extended_kalman_filter(self, mu, cov, u_t, z_t):
        # Prediction step
        mu_t_bar = self.g_nonlin_motion(mu, u_t)
        G = self.G_nonlin_motion(mu, u_t)
        sigma_t_bar = G @ cov @ G.T + self.R

        # Observation prediction
        h = self.h_observation(mu_t_bar)
        H = self.H_observation()

        # Kalman Gain calculation
        K = sigma_t_bar @ H.T @ np.linalg.inv(H @ sigma_t_bar @ H.T + self.Q)

        # Update step
        mu = mu_t_bar + K @ (z_t - h)
        sigma = (np.eye(3) - K @ H) @ sigma_t_bar

        return mu, sigma

    def gnd_truth_dynamic(self, state_gnd, u_t):
        # This is only for visualization
        At, Bt = self.dynamic_model(state_gnd[2])
        # embed()
        xt = At@state_gnd + Bt@u_t
        # print(xt)
        return xt

# Defining pre computed trajectory for u=[v,w]'=[linear velocity, angular velocity]'
# such that the robot moves in a circular
def get_trajectory_offline(radius, steps):
    u = np.zeros((2, steps))
    for i in range(steps):
        u[0, i] = 1.0  # linear velocity
        u[1, i] = 1.0 / radius  # angular velocity

    return np.array(u)

def main():
    # Create robot instance
    radius_robot = 0.2
    initial_state = np.array([0, 0, 0])  # x, y, theta
    mu0 = initial_state
    sigma0 = np.eye(3)
    proc_noise_std = np.array([0.02, 0.02, 0.01])
    obs_noise_std = np.array([0.5, 0.5, 0.2])
    radius_circle = 5.0
    robot = Robot(radius_robot, mu0, sigma0, proc_noise_std, obs_noise_std)

    # Gnd state for visualization (not used in KF) (not a distribution)
    state_gnd = np.array([0, 0, 0])

    # Number of steps for animation
    steps = 35
    
    # # Collect positions over time
    gnd_truth_x_positions = []
    gnd_truth_y_positions = []
    noisy_x_positions = []
    noisy_y_positions = []
    filtered_x_positions = []
    filtered_y_positions = []
    ekf_filtered_x_positions = []
    ekf_filtered_y_positions = []

    ctrl_inputs = get_trajectory_offline(radius_circle, steps)
    
    # Starting Kamaan Filter code
    state = initial_state
    mu = mu0
    mu_ekf = mu0
    cov = sigma0
    cov_ekf = sigma0
    # Simulate for a full circle
    for i in range(steps):
        u_present = np.transpose(ctrl_inputs[:, i])
        # Get the ground truth state (for visualization)
        state_gnd = robot.gnd_truth_dynamic(np.transpose(state_gnd), u_present)
        gnd_truth_x, ground_truth_y, _ = state_gnd
        gnd_truth_x_positions.append(gnd_truth_x)
        gnd_truth_y_positions.append(ground_truth_y)

        # Now, call action model to get the next state
        At, Bt = robot.dynamic_model(state[2])
        state = robot.action_model(At, Bt, u_present, state)
        # Gettting the sensor reading
        z_t = robot.sensor_model(robot.C, state)
        x_noisy, y_noisy, _ = z_t
        noisy_x_positions.append(x_noisy)
        noisy_y_positions.append(y_noisy)

        # Now, call Kalman Filter
        mu, cov = robot.kalman_filter(mu, cov, u_present, z_t, At, Bt)
        filtered_x, filtered_y, _ = mu
        # print("filtered_x, filtered_y at step ", i, " : ", filtered_x, filtered_y)
        filtered_x_positions.append(filtered_x)
        filtered_y_positions.append(filtered_y)

        # Similarly, let's implement EKF
        mu_ekf, cov_ekf = robot.extended_kalman_filter(mu_ekf, cov_ekf, u_present, z_t)
        ekf_filtered_x, ekf_filtered_y, _ = mu_ekf
        ekf_filtered_x_positions.append(ekf_filtered_x)
        ekf_filtered_y_positions.append(ekf_filtered_y)

    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 15)
    ax.set_title('Robot Circular Motion Simulation')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True)
    ax.axis('equal')
    
    # Create line and scatter plot objects
    ideal_line, = ax.plot([], [], 'b-', label='Ideal Path')
    noisy_scatter, = ax.plot([], [], 'r-', label='Noisy Sensor Readings')
    filtered_line, = ax.plot([], [], 'g-', label='KF Filtered Path')
    ekf_filtered_line, = ax.plot([], [], 'm', label='EKF Filtered Path')
    ax.legend()
    
    # Animation update function
    def update(frame):
        # Update ideal path
        ideal_line.set_data(gnd_truth_x_positions[:frame+1], gnd_truth_y_positions[:frame+1])
        
        # Update noisy scatter points
        noisy_scatter.set_data(noisy_x_positions[:frame+1], noisy_y_positions[:frame+1])

        # Update filtered path
        filtered_line.set_data(filtered_x_positions[:frame+1], filtered_y_positions[:frame+1])

        # Update ekf filtered path
        ekf_filtered_line.set_data(ekf_filtered_x_positions[:frame+1], ekf_filtered_y_positions[:frame+1])
        
        return ideal_line, noisy_scatter, filtered_line, ekf_filtered_line
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=steps, 
                         interval=200,  # 1 s between frames
                         blit=True, 
                         repeat=False)
    
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()