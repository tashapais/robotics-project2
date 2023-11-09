import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import argparse
import collision_checking

# Define the arm parameters
joint_radius = 0.05
link_lengths = [0.4, 0.25]
link_widths = [0.1, 0.1]
base_position = np.array([1, 1])

def forward_kinematics(theta1, theta2):
    """
    Calculate the forward kinematics for the planar 2-joint arm.
    Args:
    - theta1: The angle of the first joint in radians.
    - theta2: The angle of the second joint in radians.
    
    Returns:
    - joint_positions: A list of tuples representing the positions (x, y) of each joint.
    - end_effector_position: A tuple representing the position (x, y) of the end effector.
    """
    # Calculate the position of the first joint (which is fixed)
    joint1_pos = base_position
    
    # Calculate the position of the second joint
    joint2_pos = joint1_pos + np.array([
        link_lengths[0] * np.cos(theta1),
        link_lengths[0] * np.sin(theta1)
    ])
    
    # Calculate the position of the end effector
    end_effector_pos = joint2_pos + np.array([
        link_lengths[1] * np.cos(theta1 + theta2),
        link_lengths[1] * np.sin(theta1 + theta2)
    ])
    
    return [joint1_pos, joint2_pos], end_effector_pos

def create_link_polygon(joint_start, joint_end, width):
    """
    Create a polygon representing a link of the arm.
    
    Args:
    - joint_start: The (x, y) starting position of the joint.
    - joint_end: The (x, y) ending position of the joint.
    - width: The width of the link.
    
    Returns:
    - A numpy array of the vertices of the polygon representing the link.
    """
    # Vector from start to end
    link_vector = np.array(joint_end) - np.array(joint_start)
    link_length = np.linalg.norm(link_vector)
    link_direction = link_vector / link_length

    # Perpendicular vector to the link
    perp_vector = np.array([-link_direction[1], link_direction[0]])

    # Vertices of the rectangle
    v1 = np.array(joint_start) + width / 2 * perp_vector
    v2 = np.array(joint_start) - width / 2 * perp_vector
    v3 = np.array(joint_end) - width / 2 * perp_vector
    v4 = np.array(joint_end) + width / 2 * perp_vector

    return np.array([v1, v2, v3, v4])

def create_joint_polygon(joint_center, radius, sides=8):
    """
    Create a polygon representing a joint of the arm.
    
    Args:
    - joint_center: The (x, y) position of the joint center.
    - radius: The radius of the joint circle.
    - sides: The number of sides for the polygon approximation of the circle.
    
    Returns:
    - A numpy array of the vertices of the polygon representing the joint.
    """
    angle = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    x = joint_center[0] + radius * np.cos(angle)
    y = joint_center[1] + radius * np.sin(angle)
    return np.vstack((x, y)).T

def create_arm_polygons(joint_positions, link_widths, joint_radius):
    """
    Create polygons for the links and joints of the robot arm.
    
    Args:
    - joint_positions: A list of tuples representing the positions (x, y) of each joint.
    - link_widths: A list of the widths of each link.
    - joint_radius: The radius of the joints.
    
    Returns:
    - A list of numpy arrays, each representing a polygon for each part of the robot arm.
    """
    arm_polygons = []

    # Create the polygons for the links
    for i in range(len(joint_positions) - 1):
        link_poly = create_link_polygon(joint_positions[i], joint_positions[i + 1], link_widths[i])
        arm_polygons.append(link_poly)

    # Create the polygons for the joints
    for pos in joint_positions:
        joint_poly = create_joint_polygon(pos, joint_radius)
        arm_polygons.append(joint_poly)

    return arm_polygons

def plot_arm_and_obstacles(arm_polygons, obstacle_polygons):
    """
    Plot the robot arm and the obstacles in the workspace.
    
    Args:
    - arm_polygons: A list of numpy arrays representing the robot arm's parts.
    - obstacle_polygons: A list of numpy arrays representing the obstacles.
    """
    fig, ax = plt.subplots(dpi=100)
    ax.set_aspect('equal')

    # Plot the obstacles
    for poly in obstacle_polygons:
        polygon = Polygon(poly, closed=True, color='red', alpha=0.5)
        ax.add_patch(polygon)

    # Plot the arm
    for poly in arm_polygons:
        polygon = Polygon(poly, closed=True, color='blue', alpha=0.8)
        ax.add_patch(polygon)

    # Setting the limits of the plot
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    plt.show()


def is_configuration_collision_free(arm_polygons, obstacle_polygons):
    # Check if any of the arm's polygons collide with the obstacles
    for arm_poly in arm_polygons:
        for obs_poly in obstacle_polygons:
            if collision_checking.collides(arm_poly, obs_poly):
                return False
    return True

def generate_random_configuration():
    theta1 = np.random.uniform(-np.pi, np.pi)
    theta2 = np.random.uniform(-np.pi, np.pi)
    return theta1, theta2

def main(map_file):
    # Load the map of polygons
    obstacle_polygons = np.load(map_file, allow_pickle=True)

    # Initialize a list to store collision-free configurations
    collision_free_configs = []
    num_configs = 5

    # Generate collision-free configurations
    while len(collision_free_configs) < num_configs:
        theta1, theta2 = generate_random_configuration()
        joint_positions, end_effector_position = forward_kinematics(theta1, theta2)
        arm_polygons = create_arm_polygons(joint_positions, link_widths, joint_radius)
        
        if is_configuration_collision_free(arm_polygons, obstacle_polygons):
            collision_free_configs.append((theta1, theta2))
            # Plot the collision-free configuration
            plot_arm_and_obstacles(arm_polygons, obstacle_polygons)
            print(f"Found collision-free configuration: θ1={theta1}, θ2={theta2}")

    print(f"Generated {num_configs} collision-free configurations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random collision-free configurations for a robot arm.')
    parser.add_argument('--map', type=str, help='Path to the map file.')
    args = parser.parse_args()
    main(args.map)
