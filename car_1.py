import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--control', type=float, nargs=2)
parser.add_argument('--start', type=float, nargs=3)
args = parser.parse_args()

# Initial state [x, y, theta]
q = np.array(args.start)

# Control input [v, omega]
u = np.array(args.control)

# Robot dimensions
length = 0.2
width = 0.1

# Time step
dt = 0.1

# Control limits
v_max = 0.5
v_min = -0.5
omega_max = np.pi / 4
omega_min = -np.pi / 4

terminate = False

def car_drive_model(q, u):
    dq = np.zeros_like(q)
    dq[0] = u[0] * np.cos(q[2]) * dt
    dq[1] = u[0] * np.sin(q[2]) * dt
    dq[2] = (u[0] / 0.2) * np.tan(u[1]) * dt
    return dq

def on_key(event):
    global u
    global terminate
    if event.key == 'x':
        terminate = True

def draw_rotated_rectangle(ax, center, width, height, angle_degrees, color='b'):
    x, y = center
    rect = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=1, edgecolor=color, facecolor='none')
    t = Affine2D().rotate_deg_around(x, y, angle_degrees) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

# Initialize plot
fig, ax = plt.subplots(figsize=(6, 6))
fig.canvas.mpl_connect('key_press_event', on_key)

plt.clf()
ax = plt.gca()
plt.xlim(0, 2)
plt.ylim(0, 2)
    
# Draw robot body
draw_rotated_rectangle(ax, [q[0], q[1]], length, width, np.degrees(q[2]))
plt.pause(1)

while not terminate:
    # Update state
    dq = car_drive_model(q, u)
    q += dq
    
    # Visualization
    plt.clf()
    ax = plt.gca()
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    
    # Draw robot body
    draw_rotated_rectangle(ax, [q[0], q[1]], length, width, np.degrees(q[2]))
    
    plt.pause(0.05)