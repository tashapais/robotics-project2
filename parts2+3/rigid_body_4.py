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

def random_valid_config(original, obstacles):
    while True:
        x = random.random() * 2
        y = random.random() * 2
        theta = random.random() * 2 * np.pi
        config = [x, y, theta]
        robot = transform_rigid_body(original, config)
        collison_candidates = broadPhase(robot, obstacles)
        collision = False
        for poly in collison_candidates:
            if collides(robot, poly):
                collision = True
        min, max = np.min(robot, axis=0), np.max(robot, axis=0)
        if min[0] < 0 or min[1] < 0 or max[0] > 2 or max[1] > 2:
            collision = True
        if not collision:
            return config

def nearest_neighbor(target, configs):
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
        rotational *= .2
        distance = (.7 * euclidian) + (.3 * rotational)
        distances[i] = distance
    return np.argmin(distances)

def collision_along_configs(start, goal, obstacles):
    original = np.array([[-.1, -.05], [.1, -.05], [.1, .05], [-.1, .05]])
    resolution = .02
    deltax = goal[0] - start[0]
    xstep = deltax * resolution
    deltay = goal[1] - start[1]
    ystep = deltay * resolution
    deltatheta = (goal[2] % (2 * np.pi)) - (start[2] % (2 * np.pi))
    if deltatheta > np.pi:
        deltatheta -= (2 * np.pi)
    if deltatheta < -np.pi:
        deltatheta += (2 * np.pi)
    deltas = [deltax, deltay, deltatheta]
    thetastep = deltatheta * resolution
    current_config = np.array(start)
    while not np.array_equal(current_config, goal):
        current_config[0] += xstep
        current_config[1] += ystep
        current_config[2] += thetastep
        for i in range(3):
            if (np.absolute(current_config[i] - start[i]) > np.absolute(deltas[i])):
                current_config[i] = goal[i]
        current_robot = transform_rigid_body(original, current_config)
        broad_list = broadPhase(current_robot, obstacles)
        collision = False
        for poly in broad_list:
            if collides(current_robot, poly):
                collision = True
        min, max = np.min(current_robot, axis=0), np.max(current_robot, axis=0)
        if min[0] < 0 or min[1] < 0 or max[0] > 2 or max[1] > 2:
            collision = True
        if collision:
            return True
    return False

plt.ion()
fig, ax = plt.subplots(dpi=100)
ax.set_aspect('equal')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

original = np.array([[-.1, -.05], [.1, -.05], [.1, .05], [-.1, .05]])

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=float, nargs=3)
parser.add_argument('--goal', type=float, nargs=3)
parser.add_argument('--map')
args = parser.parse_args()
obstacles = np.load(args.map, allow_pickle=True)

#plot obstacles
for i in range(len(obstacles)):
    o = obstacles[i]
    ax.fill(o[:, 0], o[:, 1], 'blue', alpha=.25)
fig.canvas.flush_events()
fig.canvas.draw()


configs = []
parents = []
configs.append(args.start)
parents.append(-1)


for i in range(2000):
    candidate = random_valid_config(original, obstacles)
    if (i % 5 == 0):
        candidate = args.goal
    nearest_index = nearest_neighbor(candidate, configs)
    nearest_config = configs[nearest_index]
    if not collision_along_configs(nearest_config, candidate, obstacles):
        configs.append(candidate)
        parents.append(nearest_index)
        if np.array_equal(candidate, args.goal):
            break
config_path = []
config_path.append(configs.pop())
config_parent = parents.pop()
while config_parent != -1:
    config_path.append(configs[config_parent])
    config_parent = parents[config_parent]

robot_points = transform_rigid_body(original, args.start)
goal_points = transform_rigid_body(original, args.goal)
goal_patch = plt.Polygon(goal_points, color='red', alpha=.25)
robot_patch = plt.Polygon(robot_points)
ax.add_patch(robot_patch)
ax.add_patch(goal_patch)
fig.canvas.flush_events()
fig.canvas.draw()
time.sleep(1)

config_start = config_path.pop()
config_current = np.array(config_start)
while (len(config_path) > 0):
    config_next = config_path.pop()
    config_start = np.array(config_current)
    resolution = .02
    deltax = config_next[0] - config_start[0]
    xstep = deltax * resolution
    deltay = config_next[1] - config_start[1]
    ystep = deltay * resolution
    deltatheta = (config_next[2] % (2 * np.pi)) - (config_start[2] % (2 * np.pi))
    if deltatheta > np.pi:
        deltatheta -= (2 * np.pi)
    if deltatheta < -np.pi:
        deltatheta += (2 * np.pi)
    deltas = [deltax, deltay, deltatheta]
    thetastep = deltatheta * resolution
    while not np.array_equal(config_current, config_next):
        config_current[0] += xstep
        config_current[1] += ystep
        config_current[2] += thetastep
        for i in range(3):
            if (np.absolute(config_current[i] - config_start[i]) > np.absolute(deltas[i])):
                config_current[i] = config_next[i]
        robot_patch.set_xy(transform_rigid_body(original, config_current))
        time.sleep(.02)
        fig.canvas.flush_events()
        fig.canvas.draw()
time.sleep(1.5)




