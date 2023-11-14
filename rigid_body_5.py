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

def k_nearest_neighbors(target, k, configs):
    if k > len(configs):
        k = len(configs)
    if len(configs) < 1:
        return []
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
    smallestk = np.argpartition(distances, k-1)[:k]
    return smallestk

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

class Node():
    def __init__(self, parent=None, configuration=None, adjacents=None):
        self.parent = parent
        self.configuration = configuration
        self.adjacents = adjacents
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return np.array_equal(self.configuration, other.configuration)

def heuristic(node, goal_node):
    x1 = node.configuration[0]
    y1 = node.configuration[1]
    theta1 = node.configuration[2]
    x2 = goal_node.configuration[0]
    y2 = goal_node.configuration[1]
    theta2 = goal_node.configuration[2]
    euclidian = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    rotational = np.absolute(theta2 - theta1)
    if rotational > np.pi:
        rotational = (2 * np.pi) - rotational
    rotational *= .5
    distance = (.7 * euclidian) + (.3 * rotational)
    return distance


def get_neighbors(current_node):
    neighbors = []
    for adjacent in current_node.adjacents:
        new = Node(current_node, adjacent.configuration, adjacent.adjacents)
        neighbors.append(new)
    return neighbors

def astar(start, end):
    start_node = Node(None, start.configuration, start.adjacents)
    end_node = Node(None, end.configuration, end.adjacents)

    open_list = [start_node]
    closed_list = []

    while open_list:
        current_node = min(open_list, key=lambda o: o.f)
        open_list.remove(current_node)
        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.configuration)
                current = current.parent
            return path

        neighbors = get_neighbors(current_node)
        for neighbor in neighbors:
            if neighbor in closed_list:
                continue
            neighbor.g = current_node.g + heuristic(current_node, neighbor)
            neighbor.h = heuristic(neighbor, end_node)
            neighbor.f = neighbor.g + neighbor.h

            if neighbor in open_list:
                continue
            
            open_list.append(neighbor)

original = np.array([[-.1, -.05], [.1, -.05], [.1, .05], [-.1, .05]])

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=float, nargs=3)
parser.add_argument('--goal', type=float, nargs=3)
parser.add_argument('--map')
args = parser.parse_args()
obstacles = np.load(args.map, allow_pickle=True)

config_list = []
node_list = []
for i in range(502):
    if i == 500:
        config = args.start
    elif i == 501:
        config = args.goal
    else:
        config = random_valid_config(original, obstacles)
    node = Node(None, config, [])
    neighbor_indices = k_nearest_neighbors(config, 3, config_list)
    config_list.append(config)
    node_list.append(node)
    for j in neighbor_indices:
        if not collision_along_configs(config, config_list[j], obstacles):
            node_list[i].adjacents.append(node_list[j])
            node_list[j].adjacents.append(node_list[i])

path = astar(node_list[500], node_list[501])
config_start = path.pop()
config_current = np.array(config_start)

plt.ion()
fig, ax = plt.subplots(dpi=100)
ax.set_aspect('equal')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

for i in range(len(obstacles)):
    o = obstacles[i]
    ax.fill(o[:, 0], o[:, 1], 'blue', alpha=.25)
robot_points = transform_rigid_body(original, args.start)
goal_points = transform_rigid_body(original, args.goal)
goal_patch = plt.Polygon(goal_points, color='red', alpha=.25)
robot_patch = plt.Polygon(robot_points)
ax.add_patch(robot_patch)
ax.add_patch(goal_patch)
fig.canvas.flush_events()
fig.canvas.draw()
time.sleep(1)

while (len(path) > 0):
    config_next = path.pop()
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
time.sleep(1)

