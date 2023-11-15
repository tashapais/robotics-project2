import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.animation import FuncAnimation
from shapely.geometry import LineString, Polygon
import arm_3
import arm_4
from argparse import ArgumentParser

# Constants
MAX_NODES = 1000
K_NEIGHBORS = 3
GOAL_SAMPLE_RATE = 0.05  # 5% rate of sampling the goal
INITIAL_GOAL_SAMPLE_RATE = 0.5  # Initial goal sample rate
FINAL_GOAL_SAMPLE_RATE = 0.05   # Final goal sample rate
GOAL_SAMPLE_RATE_DECAY = 0.995  # Decay factor for goal sample rate

# Arm parameters
LINK_LENGTHS = [0.4, 0.25]
LINK_WIDTHS = [0.1, 0.1]
JOINT_RADIUS = 0.05
BASE_POSITION = (1, 1)

# Function to load the map file and extract polygons
def load_obstacles(map_file):
    obstacles = np.load(map_file, allow_pickle=True)
    return obstacles

# Function to generate a random configuration
def generate_random_configuration():
    return np.random.uniform(-np.pi, np.pi, size=2)

# Function to check collision between the arm and obstacles
def check_collision(config, obstacles, random_config):
    joint_positions = forward_kinematics(config[0], config[1])
    arm_links = [LineString([joint_positions[i], joint_positions[i + 1]]) for i in range(len(joint_positions) - 1)]

    for obstacle in obstacles:
        poly = Polygon(obstacle)
        for link in arm_links:
            if poly.intersects(link):
                return True  # Collision detected
    return False  # No collision detected

def check_edge_collision(config1, config2, obstacles):
    # Create a line segment between config1 and config2
    line = LineString([config1, config2])

    # Check if this line intersects with any obstacle
    for obstacle in obstacles:
        poly = Polygon(obstacle)
        if poly.intersects(line):
            return True  # Collision detected along the edge

    return False  # No collision detected along the edge

def forward_kinematics(theta1, theta2):
    l1, l2 = LINK_LENGTHS
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return (x1, y1), (x2, y2)

# Function to extract the path from the PRM tree
def extract_prm_path(parents, goal_node):
    path = []
    current_node = goal_node
    max_iterations = len(parents)  # Or some other sensible upper limit
    iteration = 0
    while current_node is not None and iteration < max_iterations:
        path.append(current_node)
        print(f"Adding node to path: {current_node}, Parent: {parents.get(tuple(current_node), None)}")
        current_node = parents.get(tuple(current_node), None)  # Safely get parent
        iteration += 1
    return path[::-1]  # Return reversed path



# Animation function to visualize the PRM roadmap growth
def animate_prm(i, nodes, edges, ax, goal_config):
    if i == 0:
        ax.clear()
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        for edge in edges:
            node1, node2 = edge
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], 'b-')
    for node in nodes[:i+1]:
        ax.plot(node[0], node[1], 'go')
    ax.plot(goal_config[0], goal_config[1], 'gx')  # Mark the goal

def parse_args():
    parser = ArgumentParser(description='PRM Algorithm for Robot Arm')
    parser.add_argument('--start', type=float, nargs=2, required=True, help='Start configuration (theta1, theta2)')
    parser.add_argument('--goal', type=float, nargs=2, required=True, help='Goal configuration (theta1, theta2)')
    parser.add_argument('--map', type=str, required=True, help='Map file containing obstacles (e.g., "arm_polygons.npy")')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()  # Parse command line arguments
    map_file = args.map  # Get the map file from command line arguments

    # Load obstacles from the map file
    obstacles = load_obstacles(map_file)

    # Define start and goal configurations using parsed arguments
    start_config = np.array(args.start)
    goal_config = np.array(args.goal)

    # Implement PRM algorithm
    prm_nodes = [start_config]
    prm_edges = []
    parents = {tuple(start_config): None}  # Start node has no parent

    print("Starting PRM construction...")
    while len(prm_nodes) < MAX_NODES:
        # Sample a random configuration
        if np.random.rand() < GOAL_SAMPLE_RATE:
            random_config = goal_config
        else:
            random_config = generate_random_configuration()

        print("Checking for collisions...")
        # Check if the random configuration is collision-free
        if not check_collision(random_config, obstacles, random_config):
            print("Collision check passed")
            prm_nodes.append(random_config)

            # Find k-nearest neighbors and update parents
            kdtree = KDTree(np.array(prm_nodes))
            distances, indices = kdtree.query(random_config, k=K_NEIGHBORS)

            for idx in indices:
                if idx < len(prm_nodes) and not check_edge_collision(prm_nodes[idx], random_config, obstacles):
                    print(f"Checking edge between node {idx} and random node...")
                    prm_edges.append((prm_nodes[idx], random_config))
                    parents[tuple(random_config)] = prm_nodes[idx]  # Update parent

        # Check if the goal configuration is reachable
        kdtree = KDTree(np.array(prm_nodes))
        distances, indices = kdtree.query(goal_config, k=K_NEIGHBORS)
        for idx in indices:
            if not check_edge_collision(prm_nodes[idx], goal_config, obstacles):
                print(f"Adding edge between nodes {idx} and {len(prm_nodes)}")
                prm_edges.append((prm_nodes[idx], goal_config))
                parents[tuple(goal_config)] = prm_nodes[idx]  # Convert goal_config to a tuple
                break



    # Visualization for roadmap growth
    fig, ax = plt.subplots()
    ani_prm = FuncAnimation(fig, animate_prm, frames=len(prm_nodes), fargs=(prm_nodes, prm_edges, ax, goal_config), interval=100)
    plt.close(fig)  # Prevents duplicate display in Colab
    print("Visualization roadmap passed")

    # Implement a discrete search algorithm (Dijkstra's or A*) to find the path
    goal_node = goal_config
    path = extract_prm_path(parents, goal_node)
    print("Search check passed")

    # Visualization for solution path
    fig, ax = plt.subplots()
    ani_solution = FuncAnimation(fig, animate_solution, frames=len(path), fargs=(path, ax, obstacles), interval=100)
    plt.close(fig)
    print("Visualization solution passed")

    # Save animations as "arm_1.5-roadmap1.mp4" and "arm_1.5-solution1.mp4"
    ani_prm.save("arm_1.5-roadmap1.mp4", writer='ffmpeg')
    ani_solution.save("arm_1.5-solution1.mp4", writer='ffmpeg')

    plt.show()
