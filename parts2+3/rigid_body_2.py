import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as pth
import random
import sys
from scipy.spatial import ConvexHull
import argparse
import random

def rotate(origin, point, angle):
    newpointx = origin[0] + np.cos(angle) * (point[0] - origin[0]) - np.sin(angle) * (point[1] - origin[1])
    newpointy = origin[1] + np.sin(angle) * (point[0] - origin[0]) + np.cos(angle) * (point[1] - origin[1])
    return [newpointx, newpointy]

def transform_rigid_body(original, config):
    rotated = np.array([rotate([0, 0], point, config[2]) for point in original])
    center = [config[0], config[1]]
    return np.array([point + center for point in rotated])

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=float, nargs=3)
parser.add_argument('--k', type=int)
parser.add_argument('--configs')
args = parser.parse_args()
configs = np.load(args.configs, allow_pickle=True)

def k_nearest_neighbors(target, k, configs):
    x1 = target[0]
    y1 = target[1]
    theta1 = target[2] % (2 * np.pi)
    distances = np.zeros(len(configs))
    for i in range(len(configs)):
        x2 = configs[i][0]
        y2 = configs[i][1]
        theta2 = configs[i][2] % (2 * np.pi)
        euclidian = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        rotational = np.absolute(theta2 - theta1)
        if rotational > np.pi:
            rotational = (2 * np.pi) - rotational
        rotational *= .5
        distance = (.7 * euclidian) + (.3 * rotational)
        distances[i] = distance
    smallestk = np.argpartition(distances, (0, k-1))[:k]
    return configs[smallestk]

fig, ax = plt.subplots(dpi=100)
ax.set_aspect('equal')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

original = np.array([[-.1, -.05], [.1, -.05], [.1, .05], [-.1, .05]])
target = transform_rigid_body(original, args.target)
ax.fill(target[:, 0], target[:, 1], 'black')

neighbors = k_nearest_neighbors(args.target, args.k, configs)
colors = ['red', 'green', 'blue']
for i in range(args.k):
    neighbor_config = neighbors[i]
    neighbor = transform_rigid_body(original, neighbor_config)
    if i > 2:
        ax.fill(neighbor[:, 0], neighbor[:, 1], 'yellow')
    else:
        ax.fill(neighbor[:, 0], neighbor[:, 1], colors[i])

for i in range(len(configs)):
    c = configs[i]
    n = transform_rigid_body(original, c)
    ax.fill(n[:, 0], n[:, 1], 'blue', alpha=.1)
plt.show()