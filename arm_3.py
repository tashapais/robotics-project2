import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

# Assuming calculate_arm_positions is defined in arm_2.py
from arm_2 import calculate_arm_positions

# Function to interpolate between two configurations
def interpolate(start_config, end_config, steps=100):
    return [start_config + (end_config - start_config) * t for t in np.linspace(0, 1, steps)]

# Function to create and save an animation
def create_animation(start_config, end_config, filename):
    # Interpolate between the start and end configurations
    path = interpolate(np.array(start_config), np.array(end_config))

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Initialize the plot elements
    arm_line, = ax.plot([], [], 'o-', lw=2)

    def init():
        arm_line.set_data([], [])
        return arm_line,

    def animate(i):
        config = path[i]
        arm_positions = calculate_arm_positions(config)
        arm_line.set_data(arm_positions[0], arm_positions[1])  # Update arm line data
        return arm_line,

    ani = animation.FuncAnimation(fig, animate, frames=len(path), init_func=init, blit=True)
    ani.save(filename, writer='ffmpeg')

# Parse arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--start', nargs=2, type=float, help='Start configuration')
parser.add_argument('--goal', nargs=2, type=float, help='Goal configuration')
args = parser.parse_args()

# Create and save the animation
create_animation(args.start, args.goal, "arm_1.3-1.mp4")