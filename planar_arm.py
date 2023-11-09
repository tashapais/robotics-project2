import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from collision_checking import collides

# Parameters
LINK_1 = 0.4
LINK_2 = 0.25
WIDTH = 0.1
RADIUS = 0.05

angle1 = 0  # global angle of joint 1
angle2 = 0  # global angle of joint 2
joint_0 = [1, 1]  # origin joint
polygons = np.load("arm_polygons.npy", allow_pickle=True)

def update_joints():
    global joint_1, joint_2
    joint_1 = [joint_0[0] + LINK_1 * math.cos(angle1),
               joint_0[1] + LINK_1 * math.sin(angle1)]
    joint_2 = [joint_1[0] + LINK_2 * math.cos(angle1 + angle2),
               joint_1[1] + LINK_2 * math.sin(angle1 + angle2)]
    return joint_1, joint_2

# Arm and Joint Configuration
joint_0 = [1, 1]
joint_1 = [joint_0[0] + LINK_1, joint_0[1]]
joint_2 = [joint_1[0] + LINK_2, joint_1[1]]

# Check and remove collision
polygons_to_remove = []
for i, poly in enumerate(polygons):
    for p in poly:
        dist1 = math.sqrt((p[0]-joint_1[0])**2 + (p[1]-joint_1[1])**2)
        dist2 = math.sqrt((p[0]-joint_2[0])**2 + (p[1]-joint_2[1])**2)
        if dist1 < LINK_1 or dist2 < LINK_2:
            polygons_to_remove.append(i)
            break

# Remove colliding polygons from the scene
for i in sorted(polygons_to_remove, reverse=True):
    polygons = np.delete(polygons, i, axis=0)

# Initialize Plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

# Plot Polygons
for poly in polygons:
    polygon = patches.Polygon(poly, edgecolor='black', facecolor='none')
    ax.add_patch(polygon)

def draw_arm(joint_0, joint_1, joint_2):
    while len(ax.lines) > 0:
        ax.lines[0].remove()

    # Draw Arm
    plt.plot([joint_0[0], joint_1[0]], [joint_0[1], joint_1[1]], 'bo-')
    plt.plot([joint_1[0], joint_2[0]], [joint_1[1], joint_2[1]], 'bo-')


def on_press(event):
    print(f"Key pressed: {event.key}")
    global angle1, angle2
    prev_angle1 = angle1
    prev_angle2 = angle2

    if event.key == 'w':
        angle1 += 0.1
    elif event.key == 'z':
        angle1 -= 0.1
    elif event.key == 'a':
        angle2 -= 0.1
    elif event.key == 'd':
        angle2 += 0.1

    joint_1, joint_2 = update_joints()
    
    # Define rectangles representing each arm segment
    arm_segment_1 = np.array([joint_0, joint_1])
    arm_segment_2 = np.array([joint_1, joint_2])

    for poly in polygons:
        # Check collision of both segments with each polygon
        if collides(arm_segment_1, poly) or collides(arm_segment_2, poly):
            # Collision detected: revert angles
            angle1 = prev_angle1
            angle2 = prev_angle2
            update_joints()
            break

    # Update the arm drawing
    draw_arm(joint_0, joint_1, joint_2)
    fig.canvas.draw()


update_joints()
draw_arm(joint_0, joint_1, joint_2)
fig.canvas.mpl_connect('key_press_event', on_press)
plt.show()