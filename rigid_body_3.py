import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as pth
import random
import sys
from scipy.spatial import ConvexHull
import argparse
import time

def rotate(origin, point, angle):
    newpointx = origin[0] + np.cos(angle) * (point[0] - origin[0]) - np.sin(angle) * (point[1] - origin[1])
    newpointy = origin[1] + np.sin(angle) * (point[0] - origin[0]) + np.cos(angle) * (point[1] - origin[1])
    return [newpointx, newpointy]

def transform_rigid_body(original, config):
    rotated = np.array([rotate([0, 0], point, config[2]) for point in original])
    center = [config[0], config[1]]
    return np.array([point + center for point in rotated])


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=float, nargs=3)
parser.add_argument('--goal', type=float, nargs=3)
args = parser.parse_args()

plt.ion()
fig, ax = plt.subplots(dpi=100)
ax.set_aspect('equal')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

resolution = .02

deltax = args.goal[0] - args.start[0]
xstep = deltax * resolution
deltay = args.goal[1] - args.start[1]
ystep = deltay * resolution
deltatheta = (args.goal[2] % (2 * np.pi)) - (args.start[2] % (2 * np.pi))
if deltatheta > np.pi:
    deltatheta -= (2 * np.pi)
if deltatheta < -np.pi:
    deltatheta += (2 * np.pi)
deltas = [deltax, deltay, deltatheta]
thetastep = deltatheta * resolution

original = np.array([[-.1, -.05], [.1, -.05], [.1, .05], [-.1, .05]])
robot_points = transform_rigid_body(original, args.start)
robot_patch = plt.Polygon(robot_points)
ax.add_patch(robot_patch)
goal_points = transform_rigid_body(original, args.goal)
ax.fill(goal_points[:, 0], goal_points[:, 1], 'red', alpha=.25)
fig.canvas.flush_events()
fig.canvas.draw()
time.sleep(1)

current_config = np.array(args.start)
while not np.array_equal(current_config, args.goal):
    current_config[0] += xstep
    current_config[1] += ystep
    current_config[2] += thetastep
    for i in range(3):
        if (np.absolute(current_config[i] - args.start[i]) > np.absolute(deltas[i])):
            current_config[i] = args.goal[i]
    robot_patch.set_xy(transform_rigid_body(original, current_config))
    time.sleep(.05)
    fig.canvas.flush_events()
    fig.canvas.draw()
time.sleep(1)



