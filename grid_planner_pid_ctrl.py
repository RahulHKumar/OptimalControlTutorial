import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import embed
from matplotlib.patches import Rectangle
import heapq

class MazeEnvironment:
    def __init__(self, start_pos, goal_pos):
        # Define maze walls as list of [x, y, width, height]
        self.maze_width = 20
        self.maze_height = 20
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.walls = [
            # Outer walls
            [-10, -5, 20, 1],   # Bottom
            [-10, 15, 20, 1],   # Top
            [-10, -5, 1, 21],   # Left
            [9, -5, 1, 21],     # Right
            
            # Inner walls
            [-5, 0, 10, 1],     # Horizontal wall
            [0, 5, 10, 1],      # Another horizontal wall
            [-5, 5, 1, 5],      # Vertical wall
        ]

        # # save the maze as a numpy array. Obstacle is 1 and free space is 0. Padding of 1 is added to the maze
        # self.pad_size = 1
        # self.maze = np.zeros((20, 20))
        # self.maze[1:20, 1:20] = 0
        # for wall in self.walls:
        #     x, y, w, h = wall
        #     self.maze[y:y+h, x:x+w] = 1
    
    def get_maze(self):
        # Initialize maze with zeros (free space)
        maze = np.zeros((self.maze_height, self.maze_width), dtype=int)
        
        # Add padding
        maze[0, :] = 1  # Top padding
        maze[-1, :] = 1  # Bottom padding
        maze[:, 0] = 1  # Left padding
        maze[:, -1] = 1  # Right padding
        
        # Add walls inside the maze
        for wall in self.walls:
            x, y, w, h = wall
            x_idx = x + 10  # Shift to fit numpy indexing (maze origin at (0,0))
            y_idx = y + 5
            maze[y_idx:y_idx+h, x_idx:x_idx+w] = 1
        # flipping the maze to match the coordinate system
        # maze = np.flipud(maze)
        return maze

    
    def check_collision(self, x, y, radius):
        # Check if robot collides with any wall
        for wall in self.walls:
            wx, wy, ww, wh = wall
            # Expand wall boundaries by robot radius for collision check
            if (x + radius > wx and x - radius < wx + ww and 
                y + radius > wy and y - radius < wy + wh):
                return True
        return False

    def draw(self, ax):
        for wall in self.walls:
            x, y, w, h = wall
            ax.add_patch(Rectangle((x+1, y+1), w, h, facecolor='gray'))
        # Draw red rectangle for start position
        ax.add_patch(Rectangle(self.start_pos, 1, 1, facecolor='red', alpha=0.5))
        # Draw green rectangle for goal position
        ax.add_patch(Rectangle(self.goal_pos, 1, 1, facecolor='green', alpha=0.5))

class Robot:
    def __init__(self, radius, mu0, Sigma0, proc_noise_std, obs_noise_std, maze):
        # Just for the looks
        self.radius = radius
        # Adding maze to the robot
        self.maze = maze

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

        # Check for collision
        if self.maze.check_collision(xt[0], xt[1], self.radius):
            # If collision detected, stay in place
            return xt_1
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

def heuristic(a, b):
    # manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbours(current_node, maze):
    neighbours = []
    for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares
        node_position = (current_node[0] + new_position[0], current_node[1] + new_position[1])
        if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
            continue # Out of bounds. They are outside the walls
        if maze[node_position[0]][node_position[1]] != 0:
            continue # It is a wall or obstacle
        neighbours.append(node_position)
    return neighbours

def path_planning(start_pos, goal_pos, algo, env_maze):
    # algo can be Astar or Greedy or UCS or BFS or DFS
    # we shall use different priority queues functions for different algorithms
    priority_func = {
        'Astar': lambda cost, depth, heuristic: cost + heuristic,
        'Greedy': lambda cost, depth, heuristic: heuristic,
        'UCS': lambda cost, depth, heuristic: cost,
        'BFS': lambda cost, depth, heuristic: depth,
        'DFS': lambda cost, depth, heuristic: -depth
    }

    if algo not in priority_func:
        raise ValueError('Invalid algorithm. Choose from Astar, Greedy, UCS, BFS, DFS')

    # Convert start and goal positions to tuple
    start_pos = tuple(start_pos)
    goal_pos = tuple(goal_pos)

    # Initializing the priority queue
    priority_queue = []
    heapq.heappush(priority_queue, (0, 0, start_pos, [start_pos])) # cost, depth, position, path
    # The above line means initial priority is 0, 
    # cost to reach the start position is 0, 
    # position is start_pos and path is [start_pos]
    visited = set()

    while len(priority_queue) > 0:
        priority, cost, current_node, path = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == goal_pos:
            x_cord, y_cord = zip(*path)
            return x_cord, y_cord

        for neighbour in get_neighbours(current_node, env_maze):
            if neighbour not in visited:
                new_cost = cost + 1
                heuristic_val = heuristic(neighbour, goal_pos)
                depth = len(path)
                new_priority = priority_func[algo](new_cost, depth, heuristic_val)
                heapq.heappush(priority_queue, (new_priority, new_cost, neighbour, path + [neighbour]))

# Modified PID Controller Implementation with PID for both linear and angular velocity
def pid_controller(current_state, target_waypoint, 
                   v_Kp=0.5, v_Ki=0.01, v_Kd=0.1, 
                   w_Kp=1.0, w_Ki=0.01, w_Kd=0.1, 
                   prev_heading_error=0, heading_error_integral=0,
                   prev_distance_error=0, distance_error_integral=0):
    # Extract current position and orientation
    current_x, current_y, current_theta = current_state
    target_x, target_y = target_waypoint
    
    # Calculate distance to target
    dx = target_x - current_x
    dy = target_y - current_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Distance error (for linear velocity control)
    # Using target distance of 0 (we want to reach the waypoint)
    distance_error = distance
    
    # Integral term for distance
    distance_error_integral += distance_error
    
    # Derivative term for distance
    distance_error_derivative = distance_error - prev_distance_error
    
    # PID control for linear velocity
    v = (v_Kp * distance_error) + (v_Ki * distance_error_integral) + (v_Kd * distance_error_derivative)
    
    # Calculate desired heading
    desired_theta = np.arctan2(dy, dx)
    
    # Calculate heading error
    heading_error = desired_theta - current_theta
    
    # Normalize heading error to [-pi, pi] range
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
    
    # Integral term for heading
    heading_error_integral += heading_error
    
    # Derivative term for heading
    heading_error_derivative = heading_error - prev_heading_error
    
    # PID control for angular velocity
    w = (w_Kp * heading_error) + (w_Ki * heading_error_integral) + (w_Kd * heading_error_derivative)
    
    # Limit velocities for safety
    v = np.clip(v, 0, 1.2)  # Max linear velocity of 1
    
    # Return control inputs and updated error terms for next iteration
    return np.array([v, w]), heading_error, heading_error_integral, distance_error, distance_error_integral

def get_trajectory_offline(start_pos, goal_pos, algo, env_maze):
    # convert start and goal positions to maze coordinates
    start_pos = np.array([start_pos[0]+10, start_pos[1]+5])
    goal_pos = np.array([goal_pos[0]+10, goal_pos[1]+5])
    env_maze = np.fliplr(env_maze)
    x_cord, y_cord = path_planning(start_pos, goal_pos, algo, env_maze)
    # convert x_cord and y_cord to actual coordinates
    x_cord = [x-10 for x in x_cord]
    y_cord = [y-5 for y in y_cord]
    return x_cord, y_cord

def main():
    # Create maze instance
    start_pos = np.array([-6, -2])
    goal_pos = np.array([6, 12])
    maze = MazeEnvironment(start_pos, goal_pos)
    env_maze= maze.get_maze()
    
    # Create robot instance
    radius_robot = 0.2
    initial_state = np.array([-6, -2, 0])  # x, y, theta
    mu0 = initial_state
    sigma0 = np.eye(3)
    proc_noise_std = np.array([0.02, 0.02, 0.01])
    obs_noise_std = np.array([0.5, 0.5, 0.2])
    radius_circle = 5.0
    robot = Robot(radius_robot, mu0, sigma0, proc_noise_std, obs_noise_std, maze)

    # Gnd state for visualization (not used in KF) (not a distribution)
    state_gnd = initial_state

    # Get path planning waypoints
    algo = 'Astar'
    x_cord, y_cord = get_trajectory_offline(start_pos, goal_pos, algo, env_maze)
    waypoints = list(zip(x_cord, y_cord))
    steps = len(waypoints)
    
    # Collect positions over time
    gnd_truth_x_positions = []
    gnd_truth_y_positions = []
    noisy_x_positions = []
    noisy_y_positions = []
    filtered_x_positions = []
    filtered_y_positions = []
    ekf_filtered_x_positions = []
    ekf_filtered_y_positions = []

    # Starting Kalman Filter code
    state = initial_state
    mu = mu0
    mu_ekf = mu0
    cov = sigma0
    cov_ekf = sigma0
    
    # Initialize PID error tracking
    prev_heading_error = 0
    heading_error_integral = 0
    prev_distance_error = 0
    distance_error_integral = 0
    
    # Simulate for each waypoint
    for i in range(1, steps):  # Start from 1 to skip the starting position
        # Call PID controller to get control inputs for current waypoint
        current_state = np.array([mu_ekf[0], mu_ekf[1], mu_ekf[2]])  # Use filtered state for control
        target_waypoint = waypoints[i]
        
        u_present, prev_heading_error, heading_error_integral, prev_distance_error, distance_error_integral = pid_controller(
            current_state, 
            target_waypoint,
            v_Kp=0.5,    # Adjust for linear velocity control
            v_Ki=0.01,   # Small integral gain to avoid overshoot
            v_Kd=0.1,    # Derivative gain for smoother response
            w_Kp=0.8,    # Proportional gain for angular velocity
            w_Ki=0.01,   # Small integral gain for heading
            w_Kd=0.1,    # Derivative gain for angular velocity
            prev_heading_error=prev_heading_error,
            heading_error_integral=heading_error_integral,
            prev_distance_error=prev_distance_error,
            distance_error_integral=distance_error_integral
        )
        
        # Get the ground truth state (for visualization)
        state_gnd = robot.gnd_truth_dynamic(state_gnd, u_present)
        # gnd_truth_x, ground_truth_y, _ = state_gnd
        # gnd_truth_x_positions.append(gnd_truth_x)
        # gnd_truth_y_positions.append(ground_truth_y)

        # Now, call action model to get the next state
        At, Bt = robot.dynamic_model(state[2])
        state = robot.action_model(At, Bt, u_present, state)
        gnd_truth_x, ground_truth_y, _ = state # from action model
        gnd_truth_x_positions.append(gnd_truth_x)
        gnd_truth_y_positions.append(ground_truth_y)
        
        # Getting the sensor reading
        z_t = robot.sensor_model(robot.C, state)
        x_noisy, y_noisy, _ = z_t
        noisy_x_positions.append(x_noisy)
        noisy_y_positions.append(y_noisy)

        # Now, call Kalman Filter
        mu, cov = robot.kalman_filter(mu, cov, u_present, z_t, At, Bt)
        filtered_x, filtered_y, _ = mu
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
    ax.set_title('Robot path planning in maze')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True)
    ax.set_xticks(np.arange(-10, 11, 1))
    ax.set_yticks(np.arange(-5, 16, 1))
    ax.axis('equal')

    # Draw maze
    maze.draw(ax)
    
    # Create line and scatter plot objects
    ideal_line, = ax.plot([], [], 'b-', label='Ground truth')
    noisy_scatter, = ax.plot([], [], 'r-', label='Noisy Sensor Readings')
    filtered_line, = ax.plot([], [], 'g-', label='KF Filtered Path')
    ekf_filtered_line, = ax.plot([], [], 'm', label='EKF Filtered Path')
    robot_dot = ax.plot([], [], 'ko', markersize=2*robot.radius)[0]

    # Drawing the path found by path planning algorithm (saved in x_cord, y_cord)
    ax.plot(x_cord, y_cord, 'y--', label='Path found by '+algo)
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

        # Update robot position
        if frame < len(filtered_x_positions):
            robot_dot.set_data([filtered_x_positions[frame]], [filtered_y_positions[frame]])
        
        return ideal_line, noisy_scatter, filtered_line, ekf_filtered_line, robot_dot
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(filtered_x_positions), 
                         interval=200,  # 200 ms between frames
                         blit=True, 
                         repeat=False)
    
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()