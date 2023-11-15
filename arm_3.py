import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from argparse import ArgumentParser

# Arm parameters
LINK_LENGTHS = [0.4, 0.25]
LINK_WIDTHS = [0.1, 0.1]
JOINT_RADIUS = 0.05
BASE_POSITION = (1, 1)
# Assuming 'forward_kinematics' and 'plot_arm' functions are defined as in the previous parts

# Function to calculate the position of the second joint and the end-effector
def forward_kinematics(theta1, theta2):
    x0, y0 = BASE_POSITION
    x1 = x0 + LINK_LENGTHS[0] * np.cos(theta1)
    y1 = y0 + LINK_LENGTHS[0] * np.sin(theta1)
    x2 = x1 + LINK_LENGTHS[1] * np.cos(theta1 + theta2)
    y2 = y1 + LINK_LENGTHS[1] * np.sin(theta1 + theta2)
    return (x0, y0), (x1, y1), (x2, y2)


# Updated plotting function to take an axes object
def plot_arm(theta1, theta2, ax, color='blue'):
    (x0, y0), (x1, y1), (x2, y2) = forward_kinematics(theta1, theta2)
    # Plot base
    ax.plot(x0, y0, 'ko', markersize=JOINT_RADIUS*100)  # Base joint
    # Plot links
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=LINK_WIDTHS[0]*10)  # First link
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=LINK_WIDTHS[1]*10)  # Second link
    # Plot joints
    ax.plot(x1, y1, 'ko', markersize=JOINT_RADIUS*100)  # First joint
    ax.plot(x2, y2, 'ko', markersize=JOINT_RADIUS*100)  # End-effector

# Function to interpolate between two configurations
def interpolate_configs(start_config, goal_config, num_steps=100):
    # Generate a sequence of configurations from start to goal
    configs = [start_config + (step/num_steps)*(goal_config - start_config) for step in range(num_steps + 1)]
    return configs

# Animation function
def animate(i, configs, ax):
    ax.clear()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal', adjustable='box')
    theta1, theta2 = configs[i]
    plot_arm(theta1, theta2, ax)

def parse_args():
    parser = ArgumentParser(description='Robot Arm Animation')
    parser.add_argument('--start', type=float, nargs=2, required=True, help='Start configuration (theta1, theta2)')
    parser.add_argument('--goal', type=float, nargs=2, required=True, help='Goal configuration (theta1, theta2)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    start_config = np.array(args.start)
    goal_config = np.array(args.goal)

    configs = interpolate_configs(start_config, goal_config, num_steps=100)

    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, animate, frames=len(configs), fargs=(configs, ax), interval=100)

    # Save the animation
    video_path = 'arm_motion.mp4'
    ani.save(video_path, writer='ffmpeg', fps=10)

    # Display the animation in the notebook
    plt.close(fig)

    # Print the path to the saved video
    print(f"Video saved at: {video_path}")