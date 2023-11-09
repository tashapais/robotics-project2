import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
        self.plot()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def transformation_matrix(self, theta, length):
        return np.array([
            [np.cos(theta), -np.sin(theta), length * np.cos(theta)],
            [np.sin(theta), np.cos(theta), length * np.sin(theta)],
            [0, 0, 1]
        ])
    
    # transformation matrix approach
    def update_points(self):
        point = np.array([0, 0, 1]).reshape(3, 1)
        prev_trans = np.identity(3) # Initialize as identity matrix
        for i in range(self.n_links):
            trans = self.transformation_matrix(self.joint_angles[i], self.link_lengths[i])
            prev_trans = prev_trans @ trans
            new_point = prev_trans @ point
            new_point[0, 0] += self.points[0][0]
            new_point[1, 0] += self.points[0][1]
            self.points[i + 1][0] = new_point[0, 0]
            self.points[i + 1][1] = new_point[1, 0]

    # geometric approach
    # def update_points(self):
    #     for i in range(1, self.n_links + 1):
    #         self.points[i][0] = self.points[i - 1][0] + self.link_lengths[i - 1] * np.cos(np.sum(self.joint_angles[:i]))
    #         self.points[i][1] = self.points[i - 1][1] + self.link_lengths[i - 1] * np.sin(np.sum(self.joint_angles[:i]))
    #     self.end_effector = np.array(self.points[self.n_links]).T
    #     print(f"End effector: {self.end_effector}")
    

    def rotate_joint(self, joint_idx, direction):
        """Rotate joint by a given direction. Positive for counterclockwise, negative for clockwise."""
        delta_angle = 5 * np.pi / 180  # 5 degrees in radians
        self.joint_angles[joint_idx] += delta_angle * direction
        self.update_points()

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
    
    def on_key(self, event):
        if event.key == 'q':
            self.terminate = True
            plt.close()
            return
        elif event.key == 'z':
            self.rotate_joint(0, -1)
        elif event.key == 'x':
            self.rotate_joint(0, 1)
        elif event.key == 'c':
            self.rotate_joint(1, -1)
        elif event.key == 'v':
            self.rotate_joint(1, 1)
        self.plot()
    
    def run(self):
        while not self.terminate:
            plt.pause(0.1)

    def plot(self):
        self.ax.clear()

        for i in range(self.n_links):
            rectangle = self.draw_rectangle(self.points[i], self.points[i + 1])
            self.ax.plot(rectangle[:, 0], rectangle[:, 1], 'black')
            self.ax.fill(rectangle[:, 0], rectangle[:, 1], 'black', alpha=0.3)  # Filling the rectangle

        for i in range(self.n_links + 1):
            circle = patches.Circle(self.points[i], radius=self.joint_radius, facecolor='black')
            self.ax.add_patch(circle)

            # # Draw local axes at each joint
            # angle_cumulative = np.sum(self.joint_angles[:i+1])
            # length_axis = 0.15  # Length of the axis arrows
            # self.ax.arrow(self.points[i][0], self.points[i][1],
            #             length_axis * np.cos(angle_cumulative), length_axis * np.sin(angle_cumulative),
            #             head_width=0.02, head_length=0.05, fc='red', ec='red')  # X-axis
            # self.ax.arrow(self.points[i][0], self.points[i][1],
            #             -length_axis * np.sin(angle_cumulative), length_axis * np.cos(angle_cumulative),
            #             head_width=0.02, head_length=0.05, fc='g', ec='g')  # Y-axis

        self.ax.set_xlim([0, 2])
        self.ax.set_ylim([0, 2])
        plt.draw()
        plt.pause(1e-5)


if __name__ == "__main__":

    arm = NLinkArm([0.4, 0.25], [0.0, 0.0], joint_radius=0.05, link_width=0.1)
    
    arm.run()
