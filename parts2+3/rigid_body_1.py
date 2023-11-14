import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as pth
import random
import sys
from scipy.spatial import ConvexHull
import argparse
import random


def broadPhase(A, polygons):
    maybeCollision = []
    min1, max1 = np.min(A, axis=0), np.max(A, axis=0)
    for polygon in polygons:
        if not np.array_equal(A, polygon):
            min2, max2 = np.min(polygon, axis=0), np.max(polygon, axis=0)
            collide_x = min1[0] < max2[0] and max1[0] > min2[0]
            collide_y = min1[1] < max2[1] and max1[1] > min2[1]
            if collide_x and collide_y:
                maybeCollision.append(polygon)
    return maybeCollision

def collides(poly, poly2):
    minkowski_diff = np.array([a - b for a in poly for b in poly2])
    hull = ConvexHull(minkowski_diff)
    minkowski_hull = minkowski_diff[hull.vertices]
    path = pth.Path(minkowski_hull)
    origin = (0, 0)
    contains_origin = path.contains_point(origin)
    return contains_origin

def rotate(origin, point, angle):
    newpointx = origin[0] + np.cos(angle) * (point[0] - origin[0]) - np.sin(angle) * (point[1] - origin[1])
    newpointy = origin[1] + np.sin(angle) * (point[0] - origin[0]) + np.cos(angle) * (point[1] - origin[1])
    return [newpointx, newpointy]

def transform_rigid_body(original, center, theta):
    rotated = np.array([rotate([0, 0], point, theta) for point in original])
    return np.array([point + center for point in rotated])

def random_valid_config(original, obstacles):
    while True:
        x = random.random() * 2
        y = random.random() * 2
        center = [x, y]
        theta = random.random() * 2 * np.pi
        robot = transform_rigid_body(original, [x, y], theta)
        collison_candidates = broadPhase(robot, obstacles)
        collision = False
        for poly in collison_candidates:
            if collides(robot, poly):
                collision = True
        min, max = np.min(robot, axis=0), np.max(robot, axis=0)
        if min[0] < 0 or min[1] < 0 or max[0] > 2 or max[1] > 2:
            collision = True
        if not collision:
            return robot

parser = argparse.ArgumentParser()
parser.add_argument('--map')
args = parser.parse_args()
obstacles = np.load(args.map, allow_pickle=True)

fig, ax = plt.subplots(dpi=100)
ax.set_aspect('equal')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

original = np.array([[-.1, -.05], [.1, -.05], [.1, .05], [-.1, .05]])
random_configs = []

while len(random_configs) < 5:
    random_configs.append(random_valid_config(original, obstacles))

for poly in obstacles:
    a = plt.Polygon(poly, fill=False, edgecolor='black', linewidth=.5, alpha=.5)
    ax.add_patch(a)
for config in random_configs:
    robot_patch = plt.Polygon(config, fill=True, edgecolor='red', linewidth=.5, alpha=.5)
    ax.add_patch(robot_patch)
plt.show()
