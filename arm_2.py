# Add imports and other necessary functions from arm_1.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import collision_checking


# Existing NLinkArm class and other functions from arm_1.py
class NLinkArm(object):
    """
    Class for controlling and plotting a planar arm with an arbitrary number of links.
    """

    def __init__(self, link_lengths, joint_angles, joint_radius, link_width):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.joint_radius = joint_radius
        self.link_width = link_width
        self.points = np.ones((self.n_links + 1, 2))

        self.terminate = False
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim([0, 2])
        self.ax.set_ylim([0, 2])
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_points()
        self.plot(color='green')

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    # geometric approach
    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + self.link_lengths[i - 1] * np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + self.link_lengths[i - 1] * np.sin(np.sum(self.joint_angles[:i]))
        self.end_effector = np.array(self.points[self.n_links]).T
    

    def rotate_joint(self, joint_idx, direction, obstacles):
        # Store the original angles in case we need to revert
        original_angles = np.copy(self.joint_angles)

        # Rotate the joint
        delta_angle = 5 * np.pi / 180  # 5 degrees in radians
        self.joint_angles[joint_idx] += delta_angle * direction
        self.update_points()

        # Check for collision with each obstacle
        collision_detected = False
        for obstacle in obstacles:
            if self.check_collision_with_obstacle(obstacle):
                collision_detected = True
                break

        if collision_detected:
            # Revert to the original joint angles in case of collision
            self.joint_angles = original_angles
            self.update_points()
            print("Collision detected, reverting to previous state.")




    def draw_rectangle(self, start, end):
        """Create a rectangle from start to end with a certain width."""
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        direction = direction / length

        # Adjust the start and end points to account for the circle radius
        start_adj = start + self.joint_radius * direction
        end_adj = end - self.joint_radius * direction

        # Calculate the perpendicular direction
        perp_direction = np.array([-direction[1], direction[0]])
        half_width_vec = 0.3 * self.link_width * perp_direction

        # Calculate the 4 corners of the rectangle
        p1 = start_adj - half_width_vec
        p2 = start_adj + half_width_vec
        p3 = end_adj + half_width_vec
        p4 = end_adj - half_width_vec

        return np.array([p1, p4, p3, p2, p1])
    
    def on_key(self, event, obstacles):
        if event.key == 'q':
            self.terminate = True
            plt.close()
            return
        elif event.key == 'z':
            self.rotate_joint(0, -1, obstacles)
        elif event.key == 'x':
            self.rotate_joint(0, 1, obstacles)
        elif event.key == 'c':
            self.rotate_joint(1, -1, obstacles)
        elif event.key == 'v':
            self.rotate_joint(1, 1, obstacles)
        self.plot(obstacles, color='black')

    
    def run(self):
        while not self.terminate:
            plt.pause(0.1)

    def plot(self, obstacles=None, color='green'):
        self.ax.clear()

        # Draw the arm
        for i in range(self.n_links):
            rectangle = self.draw_rectangle(self.points[i], self.points[i + 1])
            self.ax.plot(rectangle[:, 0], rectangle[:, 1], color)
            self.ax.fill(rectangle[:, 0], rectangle[:, 1], color, alpha=0.3)

        for i in range(self.n_links + 1):
            circle = patches.Circle(self.points[i], radius=self.joint_radius, facecolor=color)
            self.ax.add_patch(circle)

        # Draw obstacles if provided
        if obstacles is not None:
            for obstacle in obstacles:
                polygon = patches.Polygon(obstacle, edgecolor='black', facecolor='none')
                self.ax.add_patch(polygon)

        self.ax.set_xlim([0, 2])
        self.ax.set_ylim([0, 2])
        plt.draw()
        plt.pause(1e-5)

    @staticmethod
    def circle_to_polygon(center, radius, num_vertices=8):
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        return np.array([[center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)] for angle in angles])

    def check_collision_with_obstacle(self, obstacle):
        # Check for each link
        for i in range(self.n_links):
            rectangle = self.draw_rectangle(self.points[i], self.points[i + 1])
            arm_link_polygon = patches.Polygon(rectangle).get_path().to_polygons()[0]
            if collision_checking.collides(np.array(arm_link_polygon), np.array(obstacle)):
                return True

        # Check for each joint
        for point in self.points:
            joint_polygon = NLinkArm.circle_to_polygon(point, self.joint_radius)
            if collision_checking.collides(np.array(joint_polygon), np.array(obstacle)):
                return True

        return False

# New code specific to arm_2.py
def load_configurations(file_path):
    return np.load(file_path)

def compute_distance(config1, config2):
    return np.linalg.norm(np.array(config1) - np.array(config2))

def find_nearest_neighbors(target, configurations, k):
    distances = [(compute_distance(target, config), config) for config in configurations]
    distances.sort(key=lambda x: x[0])
    return [x[1] for x in distances[:k]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot nearest neighbor robot configurations.')
    parser.add_argument('--target', nargs=2, type=float, help='Target configuration [angle1, angle2]')
    parser.add_argument('-k', type=int, help='Number of nearest neighbors to find')
    parser.add_argument('--configs', type=str, help='File path to robot configurations')
    args = parser.parse_args()

    target_config = args.target
    configurations = load_configurations(args.configs)
    nearest_neighbors = find_nearest_neighbors(target_config, configurations, args.k)

    # Create a single instance of NLinkArm
    arm = NLinkArm([0.4, 0.25], target_config, joint_radius=0.05, link_width=0.1)

    # Define colors for plotting
    colors = ['red', 'green', 'blue'] + ['yellow'] * (args.k - 3)

    plt.figure()
    arm.update_joints(target_config)
    arm.plot(color='black')  # Plot the target arm in black

    # Plot the nearest neighbors
    for i, config in enumerate(nearest_neighbors):
        arm.update_joints(config)
        arm.plot(color=colors[min(i, len(colors)-1)])

    plt.show()

