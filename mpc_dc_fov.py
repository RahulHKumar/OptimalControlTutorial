import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
from sklearn.cluster import DBSCAN, KMeans

# Parameters
T = 0.05  # Time step
N = 10   # Prediction horizon (increased for better planning)
obs_x, obs_y, obs_r = 3.5, 3, 0.75  # Obstacle 1 (center and radius)
obs2_x, obs2_y, obs2_r = 1, 1.0, 0.5  # Obstacle 2 (center and radius)

# Store obstacles in a list
obstacles = [(obs_x, obs_y, obs_r), (obs2_x, obs2_y, obs2_r), (1, 4.5, 1.0), (3, 0, 0.5)]

reached_goal_threshold = 0.1  # Threshold to stop the simulation

goal_x, goal_y = 5, 5  # Goal position
max_iters = 100  # Max iterations to reach the goal
safety_margin = 0.1  # Safety margin for obstacle avoidance

# sensor parameters
fov = 120  # Field of view in degrees
sensor_range = 2  # Sensor range in meters
num_ray = 200 # Resolution of the sensor
resolution = 0.01 # Resolution of the sensor in meters

def cluster_black_pixels(image, method="dbscan", eps=5, min_samples=5, n_clusters=3):
    gray = image[:, :, 0]  # Use only the first channel
    black_pixels = np.column_stack(np.where(gray == 0))

    if len(black_pixels) == 0:
        return []  # No black pixels found

    if method.lower() == "dbscan":
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(black_pixels)

    elif method.lower() == "kmeans":
        if len(black_pixels) < n_clusters:
            return []  # Not enough points for K-Means
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clustering.fit_predict(black_pixels)

    else:
        raise ValueError("Invalid method. Choose 'dbscan' or 'kmeans'.")

    clusters = []
    unique_labels = set(labels) - {-1}  # Remove noise (-1) from DBSCAN

    for label in unique_labels:
        cluster_points = black_pixels[labels == label]  # Get points belonging to the cluster

        if len(cluster_points) < min_samples:
            continue  # Skip tiny clusters

        # Fit a minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cluster_points.astype(np.float32))
        clusters.append((int(x), int(y), int(radius)))

    return clusters

def isObstacle(world_x, world_y):
    for obs_x, obs_y, obs_r in obstacles:
        if (world_x - obs_x)**2 + (world_y - obs_y)**2 <= obs_r**2:
            return True
    return False

# Bresenham's Line Algorithm
def bresenham(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return points

def simulate_lidar(x, y, theta):
    # Initialize sensor window with resolution
    grid_size = int(6/resolution)
    display_window = np.zeros((grid_size, grid_size))
    
    # Calculate center of grid (robot position)
    grid_center_x = grid_size // 2
    grid_center_y = grid_size // 2
    
    # Fill with background color (gray)
    display_window.fill(0.5)
    
    # For each ray in the field of view
    half_fov = fov / 2
    angle_increment = fov / (num_ray-1)
    
    for i in range(num_ray):
        # Calculate ray angle
        ray_angle = theta - np.deg2rad(half_fov) + np.deg2rad(i * angle_increment)
        
        # Calculate endpoint of ray based on sensor range (in grid coordinates)
        end_x = int(grid_center_x + (sensor_range/resolution) * np.cos(ray_angle))
        end_y = int(grid_center_y + (sensor_range/resolution) * np.sin(ray_angle))
        
        # Clamp to grid boundaries
        end_x = max(0, min(end_x, grid_size-1))
        end_y = max(0, min(end_y, grid_size-1))
        
        # Use Bresenham's algorithm to get all points along the ray
        ray_points = bresenham(grid_center_x, grid_center_y, end_x, end_y)
        
        # Mark ray points in the display window
        hit_obstacle = False
        for point_x, point_y in ray_points:
            if 0 <= point_x < grid_size and 0 <= point_y < grid_size:
                # Convert grid coordinates to world coordinates
                world_x = x + (point_x - grid_center_x) * resolution
                world_y = y + (point_y - grid_center_y) * resolution
                
                if hit_obstacle:
                    # After hitting an obstacle, mark the rest of the ray black
                    display_window[point_y, point_x] = 0.0
                else:
                    # Check if point is an obstacle
                    if isObstacle(world_x, world_y):
                        display_window[point_y, point_x] = 0.0  # Mark obstacle as black
                        hit_obstacle = True  # Set flag but don't break
                    else:
                        display_window[point_y, point_x] = 1.0  # Mark ray path as white
    
    # Mark robot position
    display_window[grid_center_y, grid_center_x] = 1.0  # Robot position as white
    
    # Convert to 0-255 range for display
    display_window = (display_window * 255).astype(np.uint8)
    
    # Convert to RGB
    display_window = np.stack([display_window]*3, axis=-1)

    # mask black pixels
    mask = np.all(display_window == [0, 0, 0], axis=-1)
    plain_white = np.ones((grid_size, grid_size))
    plain_white = np.stack([plain_white]*3, axis=-1)
    plain_white[mask] = [0, 0, 0]
    # cluster maksed image
    clusters_pixel = cluster_black_pixels(plain_white, method="dbscan", eps=5, min_samples=5)
    # cluster using kmeans
    # clusters_pixel = cluster_black_pixels(plain_white, method="kmeans", n_clusters=2)
    clusters = [None] * len(clusters_pixel)
    # convert pixel coordinates to world coordinates and store in clusters
    for i, (pixel_x, pixel_y, pixel_radius) in enumerate(clusters_pixel):
        world_x = x + (pixel_x - grid_center_x) * resolution
        world_y = y + (pixel_y - grid_center_y) * resolution
        world_radius = pixel_radius * resolution  # Scale the radius
        clusters[i] = (world_x, world_y, world_radius)
    print(clusters)
    
    return display_window, clusters_pixel
    

def solve_mpc(x0, y0, theta0):
    opti = ca.Opti()
    X = opti.variable(N+1)
    Y = opti.variable(N+1)
    Theta = opti.variable(N+1)
    V = opti.variable(N)
    Omega = opti.variable(N)
    
    # System dynamics constraints
    for k in range(N):
        opti.subject_to(X[k+1] == X[k] + V[k] * ca.cos(Theta[k]) * T)
        opti.subject_to(Y[k+1] == Y[k] + V[k] * ca.sin(Theta[k]) * T)
        opti.subject_to(Theta[k+1] == Theta[k] + Omega[k] * T)
    
    # Initial conditions
    opti.subject_to(X[0] == x0)
    opti.subject_to(Y[0] == y0)
    opti.subject_to(Theta[0] == theta0)
    
    # Obstacle avoidance constraints with safety margin
    for k in range(N+1):
        opti.subject_to((X[k] - obs_x)**2 + (Y[k] - obs_y)**2 >= (obs_r + safety_margin)**2)
        opti.subject_to((X[k] - obs2_x)**2 + (Y[k] - obs2_y)**2 >= (obs2_r + safety_margin)**2)
    
    # Control constraints
    opti.subject_to(opti.bounded(0, V, 2))  # 0 ≤ v ≤ 2
    opti.subject_to(opti.bounded(-1, Omega, 1))  # -1 ≤ ω ≤ 1
    
    # Cost function: Higher weight on reaching goal
    goal_weight = 150  # Increased goal weight
    effort_weight = 0.5  # Control effort weight
    cost = effort_weight * (ca.sumsqr(V) + ca.sumsqr(Omega)) + goal_weight * ((X[N] - goal_x)**2 + (Y[N] - goal_y)**2)
    opti.minimize(cost)
    
    # Solve
    opti.solver('ipopt')
    sol = opti.solve()
    return sol.value(V[0]), sol.value(Omega[0])

# Simulation setup
x_traj, y_traj, theta_traj = [], [], []
x, y, theta = 0, 0, 0  # Initial state

# Run MPC loop
for _ in range(max_iters):
    v_opt, omega_opt = solve_mpc(x, y, theta)  # Solve MPC
    
    # # Apply first control input with small perturbation if near obstacle
    # if np.linalg.norm([x - obs_x, y - obs_y]) < obs_r + safety_margin + 0.2 or \
    #    np.linalg.norm([x - obs2_x, y - obs2_y]) < obs2_r + safety_margin + 0.2:
    #     omega_opt += np.random.uniform(-0.1, 0.1)  # Small perturbation to escape local minima
    
    x += v_opt * np.cos(theta) * T
    y += v_opt * np.sin(theta) * T
    theta += omega_opt * T
    
    # Store trajectory
    x_traj.append(x)
    y_traj.append(y)
    theta_traj.append(theta)
    
    # Stop if close to goal
    if np.linalg.norm([x - goal_x, y - goal_y]) < reached_goal_threshold:
        break

# Plot results
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# fig, ax = plt.subplots(figsize=(6, 6))
ax2.set_title("Sensor Display")
ax.scatter([0, goal_x], [0, goal_y], color='red', label="Start/Goal")
ax.scatter(obs_x, obs_y, color='black', marker='x', label="Obstacle")
ax.scatter(obs2_x, obs2_y, color='black', marker='x', label="Obstacle")
# Plot circles for all obstacles
for obstacle in obstacles:
    circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='gray', alpha=0.5)
    ax.add_patch(circle)
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.legend()
ax.grid()
ax.set_title("MPC for Unicycle Model with Obstacles")

# Plot robot trajectory
robot, = ax.plot([], [], 'bo-', markersize=5, label="Robot")
arrow = ax.quiver([], [], [], [], scale=10, color='blue')

def update(frame):
    robot.set_data(x_traj[:frame+1], y_traj[:frame+1])
    arrow.set_offsets([[x_traj[frame], y_traj[frame]]])
    arrow.set_UVC([np.cos(theta_traj[frame])], [np.sin(theta_traj[frame])])

    # calling the sensor simulation
    sensor_output, obstacle_clusters = simulate_lidar(x_traj[frame], y_traj[frame], theta_traj[frame])
    ax2.clear()
    # plot red dot at x_traj[frame], y_traj[frame]
    ax2.scatter(sensor_output.shape[1]//2, sensor_output.shape[0]//2, color='red')
    # plot clusters
    for cluster in obstacle_clusters:
        center_x, center_y, radius = cluster
        circle = plt.Circle((center_x, center_y), radius, color='red', alpha=0.5)
        ax2.add_patch(circle)
    ax2.set_title("Sensor Display")
    ax2.imshow(sensor_output, cmap='binary')
    return robot, arrow

ani = FuncAnimation(fig, update, frames=len(x_traj), interval=200, blit=False, repeat=False)
# Save the animation as a video file (MP4 format)
ani.save("mpc_dc_simulation.mp4", writer='ffmpeg', fps=5)
plt.show()
