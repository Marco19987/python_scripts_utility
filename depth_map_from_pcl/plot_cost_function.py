import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


import pickle
import cv2
import numpy as np
import nvisii 
import math
import geometric_helper as geometric_helper
import image_helper as image_helper
import nvisii_helper as nvisii_helper



def compute_cost(resized_image1_array, resized_image2_array, real_obj_image, cad_obj_image):


    h_image,w_image = resized_image1_array.shape  

    aspect_ratio_real = real_obj_image.shape[1] / real_obj_image.shape[0]
    aspect_ratio_cad = cad_obj_image.shape[1] / cad_obj_image.shape[0]
    
    cost = 0
    depth_count = 0
    for w in range(w_image):
        for h in range(h_image):
            d1 = resized_image1_array[h][w]
            d2 = resized_image2_array[h][w]
                        
            if math.isnan(d1) and math.isnan(d2):  
                cost = cost + 0.5 # is good
            else:
                if math.isnan(d2) or math.isnan(d1):
                    cost = cost + 1
                else:
                    cost = cost + 0*pow((d1-d2),2) + 1*abs(d1-d2)
                    depth_count = depth_count + 1

    
    diff_aspect_ratio = (aspect_ratio_real - aspect_ratio_cad)**2
    cost = (cost/(h_image*w_image)) + 0*diff_aspect_ratio
    #cost = cost/(depth_count)
    # Convert the images to NumPy arrays
    # img1 = np.array(resized_image1_array)
    # img2 = np.array(resized_image2_array)

    # # Create masks for the different conditions
    # both_nan = np.isnan(img1) & np.isnan(img2)
    # one_nan = np.isnan(img1) ^ np.isnan(img2)
    # no_nan = ~(np.isnan(img1) | np.isnan(img2))

    # # Calculate the cost for each condition
    # cost_both_nan = np.sum(both_nan) * 0.5
    # cost_one_nan = np.sum(one_nan)
    # cost_no_nan = np.sum(np.abs(img1[no_nan] - img2[no_nan]))

    # # Calculate the total cost
    # total_cost = cost_both_nan + cost_one_nan + cost_no_nan

    # # Normalize the cost
    # cost = total_cost / (img1.size)
    
    
    return cost


# Initialization of intrinsic and extrinsic parameters
focal_length_x = 610.0  # Focal length in pixels (along X-axis)
focal_length_y = 610.0  # Focal length in pixels (along Y-axis)
principal_point_x = 317.0  # Principal point offset along X-axis (in pixels)
principal_point_y = 238.0  # Principal point offset along Y-axis (in pixels)

image_height = 480
image_width = 640
image_dimensions = (image_height, image_width)

camera_intrinsics = [focal_length_x,focal_length_y,principal_point_x,principal_point_y,image_width,image_height]




# Load the pre-generated viewpoints from the file
viewpoint_file = 'connectorMO6_viewpoints_45aa.pkl'
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
    
    

# Load real object file
object_name = "banana"
file_name = "cad_models/connectorMO6.obj"  
mesh_scale_real = 0.01 #0.01 banana
max_virtual_depth = 5 #[m]


############## generate depth image of the real object ####################################################

# Pose real object
translation_real = np.array([0,0,0.2]) # position of the object in meters wrt camera
euler_angles_real = [0,0,0] # radians - roll pitch and yaw

# import random
# euler_angles = [random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi)]
quaternion_real = geometric_helper.euler_to_quaternion(euler_angles_real)#[0,0.5,0.5,0]  

# initialize nvisii
interactive = False
nvisii_helper.initialize_nvisii(interactive, camera_intrinsics,object_name, file_name)

# Generate the real depth map
depth_map, object_pixels = image_helper.generate_depth_map(object_name,translation_real, quaternion_real,mesh_scale_real, camera_intrinsics, max_virtual_depth) # The first time call it two times due to nvisii bug
depth_map, object_pixels = image_helper.generate_depth_map(object_name,translation_real, quaternion_real,mesh_scale_real, camera_intrinsics, max_virtual_depth) 


# crop object image
obj_depth_image = image_helper.crop_object_image(depth_map,object_pixels)

# normalize object depth map
obj_depth_image_normalized = image_helper.normalize_depth_map(obj_depth_image)

############################################################################################################


###################### FIND INITIAL GUESS FOR THE ORIENTATION ################################################

#Use the viewpoints to find a good initial guess for the orientation

# The real aspect ratio
aspect_ratio_real = obj_depth_image.shape[1] / obj_depth_image.shape[0]

# Sort the data based on the absolute difference with the real aspect ratio
#data.sort(key=lambda x: abs(x['aspect_ratio'] - aspect_ratio_real))

# Number of elements to select
N = 5000

# Select the first N elements
selected_data = data[:N]

# Iterate over each selected element
index_min = 0
cost_min = 1000
costs = []
# for i, element in enumerate(selected_data):
#     euler_angles = element['orientation']
#     depth_map_cad = element['depth_map']
#     aspect_ratio = element['aspect_ratio']
    
#     res1, res2 = resize_images_to_same_size(obj_depth_image_normalized, depth_map_cad)
#     cost = compute_cost(res1, res2)
#     costs.append(cost)
#     if cost <= cost_min:
#         index_min = i
#         cost_min = cost
        
#     # if cost_min < 0.05:
#     #     break
    
    
#     # Print the data for this element
#     print(f'Element {i}:')
#     print(f'Euler angles: {euler_angles}')
#     print(f'Aspect ratio: {aspect_ratio}')
#     print("real aspect ratio", aspect_ratio_real)
#     print(f'Depth map shape: {depth_map.shape}')
#     print("cost", cost)    
#     print()
    # cv2.imshow('Depth map', depth_map)
    # cv2.waitKey(0)

costs = np.array([compute_cost(image_helper.resize_images_to_same_size(obj_depth_image_normalized, element['depth_map'])[0],
                               image_helper.resize_images_to_same_size(obj_depth_image_normalized, element['depth_map'])[1],obj_depth_image_normalized,element['depth_map'])
                  for element in selected_data])
cost_min = np.min(costs)
index_min = np.argmin(costs)


sort_cost = np.sort(costs)
print("First ten costs", sort_cost[:10])
cost_min = sort_cost[1]
index_min = np.where(costs == cost_min)[0][0]

print("index_min", index_min)
print("cost_min", cost_min) 
depth_cad = selected_data[index_min]['depth_map']


cv2.imshow("real object", obj_depth_image_normalized)
cv2.imshow("cad object", depth_cad)
cv2.waitKey(0)
cv2.destroyAllWindows()

orientation2 = selected_data[index_min]['orientation']
print("initial guess", orientation2)


#################### plot cost function ########################################################################
orientation_angles = np.array([element['orientation'] for element in selected_data])

fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
ax1.plot(costs)
ax2.plot(orientation_angles[:,0])
ax3.plot(orientation_angles[:,1])
ax4.plot(orientation_angles[:,2])
ax1.set_ylabel('Cost')
ax1.set_title('Cost function')
ax2.set_ylabel('Orientation 1')
ax3.set_ylabel('Orientation 2')
ax4.set_ylabel('Orientation 3')
ax4.set_xlabel('sample')
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

# plot a vertical axis in correspondence of the minimum cost
ax1.axvline(index_min, color='r', linestyle='--', label='Minimum cost')
ax2.axvline(index_min, color='r', linestyle='--', label='Minimum cost')
ax3.axvline(index_min, color='r', linestyle='--', label='Minimum cost')
ax4.axvline(index_min, color='r', linestyle='--', label='Minimum cost')

# plot a vertical axis in correspondence of the groudn truth orientation
ax2.axhline(euler_angles_real[0], color='g', linestyle='--', label='Real orientation')
ax3.axhline(euler_angles_real[1], color='g', linestyle='--', label='Real orientation')
ax4.axhline(euler_angles_real[2], color='g', linestyle='--', label='Real orientation')


plt.savefig('cost_function.png')
plt.show()




from scipy.interpolate import griddata

# Create a figure with 3 subplots (3D plots)
fig = plt.figure(figsize=(18, 6))

# Prepare data for interpolation
euler_angles = np.array([element['orientation'] for element in selected_data])
# costs = np.array([compute_cost(resize_images_to_same_size(obj_depth_image_normalized, element['depth_map'])[0],
#                                resize_images_to_same_size(obj_depth_image_normalized, element['depth_map'])[1])
#                   for element in selected_data])

# Define meshgrid resolution
grid_resolution = 100j
# Subplot 1: Fix the first orientation, vary the other two
ax1 = fig.add_subplot(131, projection='3d')
grid_x1, grid_y1 = np.mgrid[min(euler_angles[:,1]):max(euler_angles[:,1]):grid_resolution, min(euler_angles[:,2]):max(euler_angles[:,2]):grid_resolution]

unique_elements, counts = np.unique(euler_angles[:,0], return_counts=True)
element_index = np.where(unique_elements == orientation2[0])[0][0]
element = unique_elements[element_index]
indices = np.array(np.where(euler_angles[:,0] == element)).T
grid_z1 = griddata(euler_angles[indices,1:3].reshape(counts[element_index],2), costs[indices], (grid_x1, grid_y1), method='cubic')
grid_z1 = np.squeeze(grid_z1[:, :, 0])

# grid_z1 = griddata(euler_angles[:,1:3], costs, (grid_x1, grid_y1), method='cubic')
ax1.plot_surface(grid_x1, grid_y1, grid_z1, cmap='viridis', edgecolor='none')
ax1.set_title('Fixed Orientation 1')
ax1.set_xlabel('Orientation 2')
ax1.set_ylabel('Orientation 3')
ax1.set_zlabel('Cost')

# Subplot 2: Fix the second orientation, vary the others
ax2 = fig.add_subplot(132, projection='3d')
grid_x2, grid_y2 = np.mgrid[min(euler_angles[:,0]):max(euler_angles[:,0]):grid_resolution, min(euler_angles[:,2]):max(euler_angles[:,2]):grid_resolution]
unique_elements, counts = np.unique(euler_angles[:,1], return_counts=True)
element_index = np.where(unique_elements == orientation2[1])[0][0]
element = unique_elements[element_index]
indices = np.array(np.where(euler_angles[:,1] == element)).T
angles_2 = euler_angles[indices, :].reshape(counts[element_index], 3)
grid_z2 = griddata(angles_2[:,[0,2]], costs[indices], (grid_x2, grid_y2), method='cubic')
grid_z2 = np.squeeze(grid_z2[:, :, 0])

#grid_z2 = griddata(euler_angles[:,[0,2]], costs, (grid_x2, grid_y2), method='cubic')
ax2.plot_surface(grid_x2, grid_y2, grid_z2, cmap='viridis', edgecolor='none')
ax2.set_title('Fixed Orientation 2')
ax2.set_xlabel('Orientation 1')
ax2.set_ylabel('Orientation 3')
ax2.set_zlabel('Cost')

# Subplot 3: Fix the third orientation, vary the others
ax3 = fig.add_subplot(133, projection='3d')
grid_x3, grid_y3 = np.mgrid[min(euler_angles[:,0]):max(euler_angles[:,0]):grid_resolution, min(euler_angles[:,1]):max(euler_angles[:,1]):grid_resolution]
unique_elements, counts = np.unique(euler_angles[:,2], return_counts=True)
element_index = np.where(unique_elements == orientation2[2])[0][0]
element = unique_elements[element_index]
indices = np.array(np.where(euler_angles[:,2] == element)).T
grid_z3 = griddata(euler_angles[indices,0:2].reshape(counts[element_index],2), costs[indices], (grid_x3, grid_y3), method='cubic')
grid_z3 = np.squeeze(grid_z3[:, :, 0])

#grid_z3 = griddata(euler_angles[:,0:2], costs, (grid_x3, grid_y3), method='cubic')
ax3.plot_surface(grid_x3, grid_y3, grid_z3, cmap='viridis', edgecolor='none')
ax3.set_title('Fixed Orientation 3')
ax3.set_xlabel('Orientation 1')
ax3.set_ylabel('Orientation 2')
ax3.set_zlabel('Cost')

# in each subplot, plot the initial guess and the real solution
ax1.scatter(orientation2[1], orientation2[2], cost_min, color='red', s=100)
ax2.scatter(orientation2[0], orientation2[2], cost_min, color='red', s=100)
ax3.scatter(orientation2[0], orientation2[1], cost_min, color='red', s=100)

# find the nearest orientation to the real one
cost_near_real = 1000
ori_diff = []
for element in selected_data:
    ori_diff.append(np.linalg.norm(np.array(element['orientation']) - np.array(euler_angles_real)))

index_min = np.argmin(ori_diff) 
cost_near_real = costs[index_min]    

ax1.scatter(euler_angles_real[1], euler_angles_real[2], cost_near_real, color='blue', s=100)
ax2.scatter(euler_angles_real[0], euler_angles_real[2], cost_near_real, color='blue', s=100)
ax3.scatter(euler_angles_real[0], euler_angles_real[1], cost_near_real, color='blue', s=100)

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.savefig('cost_function_3d_continuous_all.png')
plt.show()



cv2.destroyAllWindows()
nvisii.deinitialize()