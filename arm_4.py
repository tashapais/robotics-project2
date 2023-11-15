import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Polygon
from argparse import ArgumentParser
import arm_3

# Constants
MAX_NODES = 1000
GOAL_SAMPLE_RATE = 0.05  # 5% rate of sampling the goal

# Arm parameters
LINK_LENGTHS = [0.4, 0.25]
LINK_WIDTHS = [0.1, 0.1]
JOINT_RADIUS = 0.05
BASE_POSITION = (1, 1)
# Function to load the map file and extract polygons
def generate_random_configuration():
    return np.random.uniform(-np.pi, np.pi, size=2)

def check_collision(node, obstacles):
    # Retrieve the positions of the arm's joints based on the given configuration
    joint_positions = forward_kinematics(node.config[0], node.config[1])
    # Create line segments for the arm's links
    arm_links = [LineString([joint_positions[i], joint_positions[i + 1]]) for i in range(len(joint_positions) - 1)]

    # Check for collision with each obstacle
    for obstacle in obstacles:
        # Create a polygon for the obstacle
        poly = Polygon(obstacle)
        for link in arm_links:
            if poly.intersects(link):
                return True  # Collision detected
    return False  # No collision detected

# Function to generate a random configuration
def generate_random_configuration():
    return np.random.uniform(-np.pi, np.pi, size=2)

# Function to find the nearest node in the tree to a random configuration
def find_nearest_node(nodes, random_config):
    closest_node = nodes[0]
    min_dist = np.linalg.norm(random_config - closest_node.config)
    for node in nodes[1:]:
        dist = np.linalg.norm(random_config - node.config)
        if dist < min_dist:
            closest_node = node
            min_dist = dist
    return closest_node


# Function to steer from the nearest node towards the random configuration
def steer_towards(nearest_node, random_config, step_size=0.1):
    direction = random_config - nearest_node.config
    distance = np.linalg.norm(direction)
    if distance > step_size:
        direction = direction / distance  # Normalize
        new_config = nearest_node.config + step_size * direction
    else:
        new_config = random_config
    return RRTNode(new_config, nearest_node)


# Function to extract the path from the RRT tree
def extract_path(goal_node):
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node.config)
        current_node = current_node.parent
    return path[::-1]  # Return reversed path


# Animation function to visualize the RRT tree growth
def animate_rrt(i, nodes, ax, goal_config):
    if i == 0:
        ax.clear()
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
    node = nodes[i]
    if node.parent:
        ax.plot([node.parent.config[0], node.config[0]], [node.parent.config[1], node.config[1]], 'r-')
    ax.plot(goal_config[0], goal_config[1], 'gx')  # Mark the goal

def forward_kinematics(theta1, theta2):
    l1, l2 = LINK_LENGTHS
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return (x1, y1), (x2, y2)

def draw_robot(config, ax, obstacles):
    # Clear previous drawings
    ax.clear()

    # Compute the forward kinematics to get joint positions
    (x1, y1), (x2, y2) = forward_kinematics(config[0], config[1])

    # Draw the robot arm: first link then second link
    ax.plot([0, x1], [0, y1], 'r-')  # First link
    ax.plot([x1, x2], [y1, y2], 'r-')  # Second link

    # Draw the joints: base, first joint, and end effector
    ax.plot(0, 0, 'ko')  # Base joint (fixed at the origin)
    ax.plot(x1, y1, 'ko')  # First joint
    ax.plot(x2, y2, 'ko')  # End effector

    # Draw the obstacles as polygons
    for obstacle in obstacles:
        polygon = plt.Polygon(obstacle, color='k', alpha=0.5)
        ax.add_patch(polygon)

    # Set the plot limits to ensure all elements are visible
    ax.set_xlim(-LINK_LENGTHS[0] - LINK_LENGTHS[1], LINK_LENGTHS[0] + LINK_LENGTHS[1])
    ax.set_ylim(-LINK_LENGTHS[0] - LINK_LENGTHS[1], LINK_LENGTHS[0] + LINK_LENGTHS[1])
    ax.set_aspect('equal')

    # Optional: Configure additional plot properties like labels, grid, etc.

    return ax,

# Now the animation function
def animate_solution(i, path, ax, obstacles):
    if i == 0:
        ax.clear()
        ax.set_xlim(0, 2)  # Set these limits to fit your scene
        ax.set_ylim(0, 2)
        # Draw obstacles if needed, assuming obstacles are list of polygons
        for obstacle in obstacles:
            polygon = plt.Polygon(obstacle, color='k')
            ax.add_patch(polygon)
    # Assume the path is a list of configurations [theta1, theta2]
    config = path[i]
    draw_robot(config, ax, obstacles)  # Pass the 'obstacles' argument

    return ax,


class RRTNode:
    def __init__(self, config, parent=None):
        self.config = np.array(config)
        self.parent = parent

def rrt(start_config, goal_config, obstacles, max_nodes=MAX_NODES):
    # Initialize the RRT with the start configuration
    nodes = [RRTNode(start_config)]

    for i in range(max_nodes):
        # With a small probability, sample the goal configuration
        if np.random.rand() < GOAL_SAMPLE_RATE:
            random_config = goal_config
        else:
            random_config = generate_random_configuration()

        # Find the nearest node to the random configuration
        nearest_node = find_nearest_node(nodes, random_config)

        # Steer from the nearest node towards the random configuration
        new_node = steer_towards(nearest_node, random_config)

        # Check if the path from nearest node to new node is collision-free
        if not check_collision(new_node, obstacles):
            nodes.append(new_node)

            # Check if the new node is close to the goal configuration
            if np.linalg.norm(new_node.config - goal_config) < 0.1:  # Assuming a threshold of 0.1 for being "close"
                return nodes, new_node  # A path is found

    # If no path is found within max_nodes iterations, return the nodes without a path
    return nodes, None

def parse_args():
    parser = ArgumentParser(description='RRT Algorithm for Robot Arm')
    parser.add_argument('--start', type=float, nargs=2, required=True, help='Start configuration (theta1, theta2)')
    parser.add_argument('--goal', type=float, nargs=2, required=True, help='Goal configuration (theta1, theta2)')
    parser.add_argument('--map', type=str, required=True, help='Map file containing obstacles (e.g., "arm_polygons.npy")')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()  # Parse command line arguments
    map_file = args.map  # Get the map file from command line arguments

    # Load obstacles from the map file
    obstacles = np.load(map_file, allow_pickle=True)

    # Rest of your code remains the same
    start_config = np.array(args.start)  # Use args.start
    goal_config = np.array(args.goal)    # Use args.goal

    nodes, goal_node = rrt(start_config, goal_config, obstacles)
    if goal_node:
        path = extract_path(goal_node)
        print("Path found:", path)

        # Animate RRT growth
        fig, ax = plt.subplots()
        ani_rrt = FuncAnimation(fig, animate_rrt, frames=len(nodes), fargs=(nodes, ax, goal_config), interval=100)
        
        # Save RRT animation as a video file
        rrt_video_path = 'rrt_animation.mp4'
        ani_rrt.save(rrt_video_path, writer='ffmpeg', fps=10)
        plt.close(fig)  # Close the figure

        # Animate arm movement along the solution path
        fig, ax = plt.subplots()
        ani_solution = FuncAnimation(fig, animate_solution, frames=len(path), fargs=(path, ax, obstacles), interval=100)
        
        # Save solution animation as a video file
        solution_video_path = 'solution_animation.mp4'
        ani_solution.save(solution_video_path, writer='ffmpeg', fps=10)
        plt.close(fig)  # Close the figure

        print("RRT animation saved at:", rrt_video_path)
        print("Solution animation saved at:", solution_video_path)
    else:
        print("No path found")
