import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp
import scipy.linalg

# Define the Node class for RRT and RRT*
class Node:
    def __init__(self, x, y, vx=0, vy=0, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.parent = parent
        self.cost = cost

class KinodynamicNode:
    def __init__(self, x, y, theta, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent
        self.cost = cost

# Simulate unicycle motion
def simulate_unicycle(node, v_x, omega, dt=0.05, steps=10):
    x, y, theta = node.x, node.y, node.theta
    for _ in range(steps):
        x += v_x * np.cos(theta) * dt
        y += v_x * np.sin(theta) * dt
        theta += omega * dt
    return KinodynamicNode(x, y, theta, parent=node, cost=node.cost + steps * dt)

# Compute Euclidean distance
def distance(n1, n2):
    return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

# Check collision (simple circle)
def is_collision(node, obstacle_center, obstacle_radius):
    return distance(node, Node(*obstacle_center)) <= obstacle_radius

# Check collision between two points (used in PRM)
def edge_collision(x1, y1, x2, y2, obstacle_center, obstacle_radius):
    steps = 10
    for i in range(steps + 1):
        t = i / steps
        x = x1 * (1 - t) + x2 * t
        y = y1 * (1 - t) + y2 * t
        if distance(Node(x, y), Node(*obstacle_center)) <= obstacle_radius:
            return True
    return False

# Extract path
def extract_path(goal):
    path = []
    node = goal
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

# PRM
def prm(start, goal, n_samples=100, k=3, obstacle_center=(5,5), obstacle_radius=1.0):
    samples = [(start.x, start.y), (goal.x, goal.y)]
    while len(samples) < n_samples:
        x, y = np.random.uniform(0, 10), np.random.uniform(0, 10)
        if not is_collision(Node(x, y), obstacle_center, obstacle_radius):
            samples.append((x, y))
    G = nx.Graph()
    for i, p1 in enumerate(samples):
        G.add_node(i, pos=p1)
        dists = [((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2, j) for j, p2 in enumerate(samples) if j != i]
        dists.sort()
        for _, j in dists[:k]:
            p2 = samples[j]
            if not edge_collision(p1[0], p1[1], p2[0], p2[1], obstacle_center, obstacle_radius):
                G.add_edge(i, j, weight=np.linalg.norm(np.array(p1) - np.array(p2)))
    try:
        path_indices = nx.shortest_path(G, source=0, target=1, weight='weight')
        path = [samples[i] for i in path_indices]
    except nx.NetworkXNoPath:
        print("No path found with PRM")
        path = []
    return samples, G, path

# PRM*
def prm_star(start, goal, n_samples=100, obstacle_center=(5,5), obstacle_radius=1.0):
    samples = [(start.x, start.y), (goal.x, goal.y)]
    while len(samples) < n_samples:
        x, y = np.random.uniform(0, 10), np.random.uniform(0, 10)
        if not is_collision(Node(x, y), obstacle_center, obstacle_radius):
            samples.append((x, y))
    G = nx.Graph()
    n = len(samples)
    r = 1.0 * np.sqrt((np.log(n)/n)) * 10
    for i, p1 in enumerate(samples):
        G.add_node(i, pos=p1)
        for j, p2 in enumerate(samples):
            if i != j and np.linalg.norm(np.array(p1)-np.array(p2)) <= r:
                if not edge_collision(p1[0], p1[1], p2[0], p2[1], obstacle_center, obstacle_radius):
                    G.add_edge(i, j, weight=np.linalg.norm(np.array(p1) - np.array(p2)))
    try:
        path_indices = nx.shortest_path(G, source=0, target=1, weight='weight')
        path = [samples[i] for i in path_indices]
    except nx.NetworkXNoPath:
        print("No path found with PRM*")
        path = []
    return samples, G, path


# RRT algorithm
def rrt(start, goal, max_iters=500, step_size=0.5, obstacle_center=(5,5), obstacle_radius=1.0):
    nodes = [start]
    for _ in range(max_iters):
        rand_x, rand_y = np.random.uniform(0, 10), np.random.uniform(0, 10)
        random_node = Node(rand_x, rand_y)
        nearest = min(nodes, key=lambda node: distance(node, random_node))
        direction = np.array([random_node.x - nearest.x, random_node.y - nearest.y])
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        direction = (direction / length) * min(step_size, length)
        new_x, new_y = nearest.x + direction[0], nearest.y + direction[1]
        new_node = Node(new_x, new_y, parent=nearest)
        if is_collision(new_node, obstacle_center, obstacle_radius):
            continue
        nodes.append(new_node)
        if distance(new_node, goal) < step_size:
            goal.parent = new_node
            nodes.append(goal)
            break
    return nodes

# RRT* algorithm
def rrt_star(start, goal, max_iters=500, step_size=0.5, search_radius=1.5, obstacle_center=(5,5), obstacle_radius=1.0):
    nodes = [start]
    for _ in range(max_iters):
        rand_x, rand_y = np.random.uniform(0, 10), np.random.uniform(0, 10)
        random_node = Node(rand_x, rand_y)
        nearest = min(nodes, key=lambda node: distance(node, random_node))
        direction = np.array([random_node.x - nearest.x, random_node.y - nearest.y])
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        direction = (direction / length) * min(step_size, length)
        new_x, new_y = nearest.x + direction[0], nearest.y + direction[1]
        new_node = Node(new_x, new_y)

        if is_collision(new_node, obstacle_center, obstacle_radius):
            continue

        near_nodes = [node for node in nodes if distance(node, new_node) <= search_radius]
        min_cost = nearest.cost + distance(nearest, new_node)
        best_parent = nearest

        for node in near_nodes:
            temp_cost = node.cost + distance(node, new_node)
            if temp_cost < min_cost:
                min_cost = temp_cost
                best_parent = node

        new_node.cost = min_cost
        new_node.parent = best_parent
        nodes.append(new_node)

        for node in near_nodes:
            if new_node.cost + distance(new_node, node) < node.cost:
                node.parent = new_node
                node.cost = new_node.cost + distance(new_node, node)

        if distance(new_node, goal) < step_size:
            goal.parent = new_node
            goal.cost = new_node.cost + distance(new_node, goal)
            nodes.append(goal)
            break

    return nodes

# Kinodynamic RRT*
# Using unicycle model (Non-linear system)
def kinodynamic_rrt_star(start, goal, max_iters=500, step_size=0.5, obstacle_center=(5,5), obstacle_radius=1.0,
                         v_range=(0.0, 1.5), omega_range=(-1.0, 1.0)):
    nodes = [start]
    for _ in range(max_iters):
        rand_x, rand_y = np.random.uniform(0, 10), np.random.uniform(0, 10)
        rand_theta = np.random.uniform(-np.pi, np.pi)
        random_node = KinodynamicNode(rand_x, rand_y, rand_theta)
        nearest = min(nodes, key=lambda node: distance(node, random_node))
        v_x = np.random.uniform(v_range[0], v_range[1])
        omega = np.random.uniform(omega_range[0], omega_range[1])
        new_node = simulate_unicycle(nearest, v_x, omega)
        if is_collision(new_node, obstacle_center, obstacle_radius):
            continue
        nodes.append(new_node)
        if distance(new_node, goal) < step_size:
            goal.parent = new_node
            nodes.append(goal)
            break
    return nodes

# LQR-RRT* using double integrator model (Linear system)
A = np.array([[1, 0, 0.05, 0], [0, 1, 0, 0.05], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.array([[0.00125, 0], [0, 0.00125], [0.05, 0], [0, 0.05]])
Q = np.diag([5, 5, 0.1, 0.1])
R = np.diag([0.7, 0.7])

def simulate_lqr_riccati(from_node, to_node, dt=0.05, horizon=10):
    x0 = np.array([from_node.x, from_node.y, from_node.vx, from_node.vy])
    xg = np.array([to_node.x, to_node.y, to_node.vx, to_node.vy])
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

    traj = []
    x = x0.copy()
    for _ in range(horizon):
        u = -K @ (x - xg)
        x = A @ x + B @ u
        traj.append(Node(x[0], x[1], x[2], x[3]))

    final_node = traj[-1]
    final_node.parent = from_node
    return final_node, traj

def simulate_lqr_cvxpy(from_node, to_node, horizon=10):
    x0 = np.array([from_node.x, from_node.y, from_node.vx, from_node.vy])
    xg = np.array([to_node.x, to_node.y, to_node.vx, to_node.vy])
    x = cp.Variable((4, horizon+1))
    u = cp.Variable((2, horizon))
    cost = 0
    constraints = [x[:, 0] == x0]
    for t in range(horizon):
        cost += cp.quad_form(x[:, t] - xg, Q) + cp.quad_form(u[:, t], R)
        constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t]]
    cost += cp.quad_form(x[:, horizon] - xg, Q)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        return None, []
    final_state = x.value[:, -1]
    final_node = Node(final_state[0], final_state[1], final_state[2], final_state[3], parent=from_node)
    traj = [Node(*x.value[:, t][:4]) for t in range(horizon + 1)]
    return final_node, traj


def lqr_rrt_star(start, goal, max_iters=500, obstacle_center=(5,5), obstacle_radius=1.0, max_dist=0.5, mode='riccati'):
    nodes = [start]
    for _ in range(max_iters):
        rand = Node(np.random.uniform(0, 10), np.random.uniform(0, 10), np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        nearest = min(nodes, key=lambda node: distance(node, rand))

        dist_to_rand = distance(nearest, rand)
        if dist_to_rand > max_dist:
            direction = np.array([rand.x - nearest.x, rand.y - nearest.y])
            direction /= np.linalg.norm(direction)
            scaled_x = nearest.x + direction[0] * max_dist
            scaled_y = nearest.y + direction[1] * max_dist
            rand = Node(scaled_x, scaled_y, rand.vx, rand.vy)

        if mode == 'cvxpy':
            result = simulate_lqr_cvxpy(nearest, rand)
        elif mode == 'riccati':
            result = simulate_lqr_riccati(nearest, rand)
        else:
            raise ValueError("Unknown LQR mode: choose 'cvxpy' or 'riccati'")

        if result is None:
            continue

        new_node, _ = result
        if distance(new_node, goal) < max_dist:
            goal.parent = new_node
            nodes.append(goal)
            break
        if is_collision(new_node, obstacle_center, obstacle_radius):
            continue
        new_node.parent = nearest
        nodes.append(new_node)
    return nodes

# Set the algorithm flag
flag = 'lqr_rrt_star'  # Choose: 'rrt', 'rrt_star', 'kinodynamic_rrt_star', 'prm', 'prm_star', 'lqr_rrt_star'

# Initialize
if flag == 'kinodynamic_rrt_star':
    start_node = KinodynamicNode(1, 1, 0)
    goal_node = KinodynamicNode(9, 9, 0)
else:
    start_node = Node(1, 1, 0, 0)
    goal_node = Node(9, 9, 0, 0)

obstacle_center = (5, 5)
obstacle_radius = 1.0

# Algorithm dispatch
if flag == 'rrt':
    nodes = rrt(start_node, goal_node, obstacle_center=obstacle_center, obstacle_radius=obstacle_radius)
    path = extract_path(goal_node)
    color = 'c'; label = 'RRT'
elif flag == 'rrt_star':
    nodes = rrt_star(start_node, goal_node, obstacle_center=obstacle_center, obstacle_radius=obstacle_radius)
    path = extract_path(goal_node)
    color = 'b'; label = 'RRT*'
elif flag == 'kinodynamic_rrt_star':
    nodes = kinodynamic_rrt_star(start_node, goal_node, obstacle_center=obstacle_center, obstacle_radius=obstacle_radius)
    path = extract_path(goal_node)
    color = 'm'; label = 'Kinodynamic RRT*'
elif flag == 'prm':
    samples, graph, path = prm(start_node, goal_node, obstacle_center=obstacle_center, obstacle_radius=obstacle_radius)
    color = 'g'; label = 'PRM'
elif flag == 'prm_star':
    samples, graph, path = prm_star(start_node, goal_node, obstacle_center=obstacle_center, obstacle_radius=obstacle_radius)
    color = 'brown'; label = 'PRM*'
elif flag == 'lqr_rrt_star':
    mode = 'cvxpy'  # Choose: 'riccati' or 'cvxpy'
    nodes = lqr_rrt_star(start_node, goal_node, obstacle_center=obstacle_center, obstacle_radius=obstacle_radius, mode=mode)
    path = extract_path(goal_node)
    color = 'orange'; label = 'LQR-RRT*'

# Plot
plt.figure(figsize=(6, 6))
plt.xlim(0, 10); plt.ylim(0, 10); plt.grid(True)
plt.gca().add_patch(plt.Circle(obstacle_center, obstacle_radius, color='black', alpha=0.5))

if flag in ['rrt', 'rrt_star', 'kinodynamic_rrt_star', 'lqr_rrt_star']:
    for node in nodes:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color=color, alpha=0.2)
elif flag in ['prm', 'prm_star']:
    pos = nx.get_node_attributes(graph, 'pos')
    for edge in graph.edges:
        p1, p2 = pos[edge[0]], pos[edge[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color, alpha=0.2)
    for p in pos.values():
        plt.plot(p[0], p[1], 'go', markersize=2)

if path:
    plt.plot(*zip(*path), color=color, linewidth=2, label=label, marker='*')

plt.plot(1, 1, 'bo', markersize=8, label='Start')
plt.plot(9, 9, 'ro', markersize=8, label='Goal')
plt.legend()
plt.title(f"Path Planning using {label}")
if flag == 'lqr_rrt_star':
    plt.title(f"Path Planning using {label} ({mode})")
plt.show()
