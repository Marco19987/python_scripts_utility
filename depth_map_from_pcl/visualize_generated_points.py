import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


import pickle
import cv2
import numpy as np
import nvisii 
import geometric_helper as geometric_helper
import image_helper as image_helper
import nvisii_helper as nvisii_helper



# Load the pre-generated viewpoints from the file
viewpoint_file = 'viewpoints_data/rubber_duck_viewpoints_30aa_d.pkl'
visualize_viewpoints = False

with open(viewpoint_file, 'rb') as f:
    data = pickle.load(f)

# Iterate over each element in the data
if visualize_viewpoints:
    for i, element in enumerate(data):
        euler_angles = element['orientation']
        depth_map = element['depth_map']
        aspect_ratio = element['aspect_ratio']

        # Print the data for this element
        print(f'Element {i}:')
        print(f'orientation: {euler_angles}')
        print(f'Aspect ratio: {aspect_ratio}')
        print(f'Depth map shape: {depth_map.shape}')
        cv2.imshow('Depth map', depth_map)
        cv2.waitKey(500)
        print()


# make a 3d plot of hte points in element['orientation']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, element in enumerate(data):
    orientation = element['orientation']
    axis, angle = geometric_helper.axis_angle_from_vector(geometric_helper.axis_angle_viewpoint(orientation[0], orientation[1], orientation[2]))
    rx = np.sin(orientation[1]) * np.cos(orientation[0]) 
    ry = np.sin(orientation[1]) * np.sin(orientation[0])
    rz = np.cos(orientation[1])
    ax.scatter(axis[0], axis[1], axis[2], c='b', marker='o')
    ax.scatter(rx, ry, rz, c='g', marker='*')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.show()

# plot sphere
# step_size = 10*np.pi/180
# phi_array = np.arange(0+step_size,np.pi-step_size, step_size)
# theta_array = np.arange(0+step_size, np.pi-step_size, step_size)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for phi in phi_array:
#     for theta in theta_array:
#         axis =geometric_helper.axis_angle_viewpoint(phi, theta, 1)

#         ax.scatter(axis[0], axis[1], axis[2], c='b', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)
# plt.show()

