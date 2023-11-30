import numpy as np
import os
import scipy.optimize
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''' This library contains all functions for visualizing the results.'''


def visualize(q_pos):

    # Set figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-1.0, 0.5)
    ax.set_zlim3d(0.0, 1.0)

    # Plot robot
    T = plot_robot(q_pos, ax)
    print("\nBaselink-end effector transform...\n")
    print(T)

    # Import and plot transformation matrices
    filepath = os.path.join(os.path.abspath(os.getcwd()), '/content/drive/My Drive/vision_based_robot_picking/data/matrix_files/baselink_camera_transformation.npy')
    if os.path.isfile(filepath):
        bc_transform = np.load(filepath, allow_pickle=True)
        print("\nBaselink-camera transform...\n")
        print(bc_transform)
        plot_frame_t(bc_transform, ax, 'C')
    else:
        print("\nNo baselink-camera transform\n")
        pass

    filepath = os.path.join(os.path.abspath(os.getcwd()), '/content/drive/My Drive/vision_based_robot_picking/data/matrix_files/baselink_target_transformation.npy')
    if os.path.isfile(filepath):
        bt_transform = np.load(filepath, allow_pickle=True)
        print("\nBaselink-target transform...\n")
        print(bt_transform)
        plot_frame_t(bt_transform, ax, 'T')
    else:
        print("\nNo baselink-target transform\n")
        pass

    # Show plot
    plt.show()
    

def robot_kinematics1(q):

    T01 = sp.Matrix([[-sp.cos(q[0]), 0, -sp.sin(q[0]), 0.050 * sp.cos(q[0])],
                     [-sp.sin(q[0]), 0, sp.cos(q[0]), 0.050 * sp.sin(q[0])],
                     [0, 1, 0, 0.457],
                     [0, 0, 0, 1]])

    T12 = sp.Matrix([[-sp.sin(q[1]), -sp.cos(q[1]), 0, -0.440 * sp.sin(q[1])],
                     [sp.cos(q[1]), -sp.sin(q[1]), 0, 0.440 * sp.cos(q[1])],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    T23 = sp.Matrix([[-sp.cos(q[2]), 0, -sp.sin(q[2]), 0.035 * sp.cos(q[2])],
                     [-sp.sin(q[2]), 0, sp.cos(q[2]), 0.035 * sp.sin(q[2])],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])

    T34 = sp.Matrix([[-sp.cos(q[3]), 0, -sp.sin(q[3]), 0],
                     [-sp.sin(q[3]), 0, sp.cos(q[3]), 0],
                     [0, 1, 0, 0.420],
                     [0, 0, 0, 1]])

    T45 = sp.Matrix([[-sp.cos(q[4]), 0, -sp.sin(q[4]), 0],
                     [-sp.sin(q[4]), 0, sp.cos(q[4]), 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])

    T56 = sp.Matrix([[sp.cos(q[5]), -sp.sin(q[5]), 0, 0],
                     [sp.sin(q[5]), sp.cos(q[5]), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    T02 = T01 * T12
    T03 = T01 * T12 * T23
    T04 = T01 * T12 * T23 * T34
    T05 = T01 * T12 * T23 * T34 * T45
    T = T01 * T12 * T23 * T34 * T45 * T56

    return [T01, T02, T03, T04, T05, T]


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    x_mean = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]
    y_mean = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]
    z_mean = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])


def plot_frame_t(t, ax, text=''):
    axis_length = 0.2
    r = t[0:3, 0:3]
    x = t[0][3]
    y = t[1][3]
    z = t[2][3]
    pose_ix = np.dot(r, np.array([axis_length, 0, 0]))
    pose_iy = np.dot(r, np.array([0, axis_length, 0]))
    pose_iz = np.dot(r, np.array([0, 0, axis_length]))
    ax.plot([x, x + pose_ix[0]], [y, y + pose_ix[1]], [z, z + pose_ix[2]], 'r', linewidth=2)
    ax.plot([x, x + pose_iy[0]], [y, y + pose_iy[1]], [z, z + pose_iy[2]], 'g', linewidth=2)
    ax.plot([x, x + pose_iz[0]], [y, y + pose_iz[1]], [z, z + pose_iz[2]], 'b', linewidth=2)
    pose_t = np.dot(r, np.array([0.3 * axis_length, 0.3 * axis_length, 0.3 * axis_length]))
    ax.text(x + pose_t[0], y + pose_t[1], z + pose_t[2], text, fontsize=11)


def plot_robot(q_, ax):

    # Set up our joint angle symbols for symbolic computation
    q = [sp.Symbol('q1'), sp.Symbol('q2'), sp.Symbol('q3'), sp.Symbol('q4'), sp.Symbol('q5'), sp.Symbol('q6')]

    # Get robot kinematics
    T_ = robot_kinematics1(q)

    # Plot world frame
    plot_frame_t(np.identity(4), ax, 'w')

    # Value to substitute
    subst = [(q[0], q_[0]), (q[1], q_[1]), (q[2], q_[2]), (q[3], q_[3]), (q[4], q_[4]), (q[5], q_[5])]

    # Plot joint frames and links
    #for i in range(0, 5):
    plot_frame_t(np.array(T_[0].subs(subst)), ax, 'j' + str(0 + 1))
    plot_frame_t(np.array(T_[1].subs(subst)), ax, 'j' + str(1 + 1))
    plot_frame_t(np.array(T_[2].subs(subst)), ax, 'j' + str(2 + 1))
    plot_frame_t(np.array(T_[5].subs(subst)), ax, 'j' + str(5 + 1))
    set_axes_equal(ax)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    return np.array(T_[-1].subs(subst))
