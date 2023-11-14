import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as pth
import random
import sys
from scipy.spatial import ConvexHull
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import argparse
import time

def rotate(origin, point, angle):
    newpointx = origin[0] + np.cos(angle) * (point[0] - origin[0]) - np.sin(angle) * (point[1] - origin[1])
    newpointy = origin[1] + np.sin(angle) * (point[0] - origin[0]) + np.cos(angle) * (point[1] - origin[1])
    return [newpointx, newpointy]

def transform_rigid_body(config):
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

def car_drive_model(q, u):
    dq = np.zeros_like(q)
    dq[0] = u[0] * np.cos(q[2]) * dt
    dq[1] = u[0] * np.sin(q[2]) * dt
    dq[2] = (u[0] / 0.2) * np.tan(u[1]) * dt
    return dq

def get_random_control():
    velocity = (random.random() * 1) - .5
    angle = (random.random() * (np.pi / 2)) - (np.pi / 4)
    return [velocity, angle]

def get_random_config(obstacles):
    while True:
        x = random.random() * 2
        y = random.random() * 2
        theta = random.random() * 2 * np.pi
        config = [x, y, theta]
        robot = transform_rigid_body(config)
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

def collides_along_control(q, u, iterations, obstacles):
    qnew = np.array(q)
    for i in range(iterations):
        dq = car_drive_model(qnew, u)
        qnew += dq
        current_car = transform_rigid_body(qnew)
        broad_list = broadPhase(current_car, obstacles)
        collision = False
        for obstacle in broad_list:
            if collides(current_car, obstacle):
                collision = True
        min, max = np.min(current_car, axis=0), np.max(current_car, axis=0)
        if min[0] < 0 or min[1] < 0 or max[0] > 2 or max[1] > 2:
            collision = True
        if collision:
            return []
    qnew[2] = qnew[2] % (2 * np.pi)
    return qnew

def in_goal_region(config, goal):
    return (np.absolute(config[0] - goal[0]) < .1
    and np.absolute(config[1] - goal[1]) < .1
    and np.absolute(config[2] - goal[2]) < .5)

def draw_rotated_rectangle(ax, center, width, height, angle_degrees, color='b'):
    x, y = center
    rect = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=1, edgecolor=color, facecolor='none')
    t = Affine2D().rotate_deg_around(x, y, angle_degrees) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

original = np.array([[-.1, -.05], [.1, -.05], [.1, .05], [-.1, .05]])
dt = .1
deltat = 2
b = 30
samples = 4000
length = 0.2
width = 0.1
#get arguements
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=float, nargs=3)
parser.add_argument('--goal', type=float, nargs=3)
parser.add_argument('--map')
args = parser.parse_args()
obstacles = np.load(args.map, allow_pickle=True)
#initialize plot

configs = []
parents = []
controls = []
configs.append(args.start)
parents.append(-1)
controls.append([])

success = False
for i in range(samples):
    random_sample = get_random_config(obstacles)
    if i > samples * .75:
        if i % 5 == 0:
            random_sample = args.goal
        deltat = 1
    else:
        if i % 10 == 0:
            random_sample = args.goal
        deltat = 2
    nearest_index = nearest_neighbor(random_sample, configs)
    control_blossom = []
    config_blossom = []
    while len(control_blossom) < b:
        random_control = get_random_control()
        control_end_config = collides_along_control(configs[nearest_index], random_control, deltat, obstacles)
        if len(control_end_config) > 0:
            control_blossom.append(random_control)
            config_blossom.append(control_end_config)
    nearest_blossom_index = nearest_neighbor(random_sample, config_blossom)
    configs.append(config_blossom[nearest_blossom_index])
    controls.append(control_blossom[nearest_blossom_index])
    parents.append(nearest_index)
    if in_goal_region(config_blossom[nearest_blossom_index], args.goal):
        success = True
        break
if success:
    print('Success')
else:
    print('Failure')

control_path = []
control_path.append(controls.pop())
control_parent = parents.pop()
while control_parent != -1:
    control_path.append(controls[control_parent])
    control_parent = parents[control_parent]

fig, ax = plt.subplots(figsize=(6, 6))

plt.clf()
ax = plt.gca()
plt.xlim(0, 2)
plt.ylim(0, 2)

for i in range(len(obstacles)):
    o = obstacles[i]
    ax.fill(o[:, 0], o[:, 1], 'blue', alpha=.25)

goal_points = transform_rigid_body(args.goal)
ax.fill(goal_points[:, 0], goal_points[:, 1], 'red', alpha=.25)

# Draw robot body
draw_rotated_rectangle(ax, [args.start[0], args.start[1]], length, width, np.degrees(args.start[2]))
plt.pause(1)

q = args.start
control_path.pop()
for i in range(len(control_path)):
    u = control_path.pop()
    for j in range(deltat):
        dq = car_drive_model(q, u)
        q += dq
        
        # Visualization
        plt.clf()
        ax = plt.gca()
        plt.xlim(0, 2)
        plt.ylim(0, 2)

        for i in range(len(obstacles)):
            o = obstacles[i]
            ax.fill(o[:, 0], o[:, 1], 'blue', alpha=.25)
        
        ax.fill(goal_points[:, 0], goal_points[:, 1], 'red', alpha=.25)
        # Draw robot body
        draw_rotated_rectangle(ax, [q[0], q[1]], length, width, np.degrees(q[2]))
        
        plt.pause(0.05)

time.sleep(1)