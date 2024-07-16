import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pickle
import cv2
import numpy as np
import nvisii 
from scipy.optimize import minimize, NonlinearConstraint
from concurrent.futures import ThreadPoolExecutor
import geometric_helper as geometric_helper
import image_helper as image_helper
import nvisii_helper as nvisii_helper
import math




def orientation_cost_function(orientation,object_name,translation_cad, mesh_scale_cad, camera_intrinsics, max_virtual_depth):


    rtheta = geometric_helper.axis_angle_viewpoint(orientation[0],orientation[1],orientation[2])
    axis, angle = geometric_helper.axis_angle_from_vector(rtheta)
    quaternion_cad = geometric_helper.axis_angle_to_quaternion(axis, angle)
        
    # change cad orientation
    depth_map_cad, object_pixels2 = image_helper.generate_depth_map(object_name,translation_cad, quaternion_cad, mesh_scale_cad, camera_intrinsics, max_virtual_depth) 
    obj_depth_image2 = image_helper.crop_object_image(depth_map_cad,object_pixels2)
    obj_depth_image2_normalized = image_helper.normalize_depth_map(obj_depth_image2)
    
    # resize images to the same size
    resized_image1_array, resized_image2_array = image_helper.resize_images_to_same_size(obj_depth_image_normalized, obj_depth_image2_normalized)
        
    cost = compute_cost(resized_image1_array, resized_image2_array)  
    
    
    print("orientation", orientation)
    print("cost value", cost)
    
    cv2.imshow("depth_map cad", depth_map_cad)
    cv2.imshow("Real image", resized_image1_array)
    cv2.imshow("Cad model", resized_image2_array)
    cv2.waitKey(0)

    indipendent_vars.append(orientation)
    indipendent_cost.append(cost)

    return cost

def compute_cost(resized_image1_array, resized_image2_array):

    h_image,w_image = resized_image1_array.shape  

   
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

    
    cost = (cost/(h_image*w_image))
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

K = np.array([[focal_length_x, 0, principal_point_x],
                                [0, focal_length_y, principal_point_y],
                                [0, 0, 1]])
camera_intrinsics = [focal_length_x,focal_length_y,principal_point_x,principal_point_y,image_width,image_height]



# Load the pre-generated viewpoints from the file
viewpoint_file = 'viewpoints_data/bowl_viewpoints_20aa.pkl'

with open(viewpoint_file, 'rb') as f:
    data = pickle.load(f)



# Load real object file
object_name_real = "banana"
file_name = "cad_models/bowl.obj"  
mesh_scale_real = 0.001 #0.01 banana
max_virtual_depth = 5 #[m]


############## generate depth image of the real object ####################################################

# Pose real object
translation_real = np.array([0,0,1]) # position of the object in meters wrt camera


import random
#phi, theta, psi = random.uniform(0, np.pi), random.uniform(0, np.pi), random.uniform(0, 2*np.pi)
phi, theta, psi = 1,1,1


orientation_real = np.array([phi, theta, psi])
rtheta = geometric_helper.axis_angle_viewpoint(phi,theta,psi)
axis, angle = geometric_helper.axis_angle_from_vector(rtheta)
quaternion_real = geometric_helper.axis_angle_to_quaternion(axis, angle)



# initialize nvisii
interactive = False
nvisii_helper.initialize_nvisii(interactive, camera_intrinsics,object_name_real, file_name)

# Generate the real depth map
depth_map, object_pixels = image_helper.generate_depth_map(object_name_real,translation_real, quaternion_real,mesh_scale_real, camera_intrinsics, max_virtual_depth) # The first time call it two times due to nvisii bug
depth_map, object_pixels = image_helper.generate_depth_map(object_name_real,translation_real, quaternion_real,mesh_scale_real, camera_intrinsics, max_virtual_depth) 

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
costs = np.array([compute_cost(image_helper.resize_images_to_same_size(obj_depth_image_normalized, element['depth_map'])[0],
                               image_helper.resize_images_to_same_size(obj_depth_image_normalized, element['depth_map'])[1])
                  for element in selected_data])
cost_min = np.min(costs)
index_min = np.argmin(costs)


sort_cost = np.sort(costs)
print("First ten costs", sort_cost[:10])
# cost_min = sort_cost[0]
# index_min = np.where(costs == cost_min)[0][0]

print("index_min", index_min)
print("cost_min", cost_min) 
depth_cad = selected_data[index_min]['depth_map']


cv2.imshow("real object", obj_depth_image_normalized)
cv2.imshow("cad object", depth_cad)
cv2.waitKey(0)
cv2.destroyAllWindows()

orientation2 = selected_data[index_min]['orientation']
print("initial guess", orientation2)


#################### OPTIMIZATION ########################################################################

# load model object
object_name_cad = "banana2"
new_object_path = "cad_models/bowl.obj"
mesh_scale_cad = mesh_scale_real*1

nvisii_helper.change_object_mesh(object_name_real, object_name_cad, new_object_path)
translation_cad = geometric_helper.compute_object_center(object_pixels, 1, camera_intrinsics)

Rinit = geometric_helper.quaternion_to_rotation_matrix(geometric_helper.euler_to_quaternion(orientation2))
Rinit = geometric_helper.normalize_rotation_matrix(Rinit)


initial_guesses = [[orientation2]]#[[0,1.57, 0]]
# debug cost variables
indipendent_vars = []
indipendent_cost = []

bnds = [(0, np.pi), (0, np.pi), (0, 2*np.pi)]
# bnds = [(orientation2[0]-0.5, orientation2[0]+0.5), (orientation2[1]-0.5, orientation2[1]+0.5), (orientation2[2]-0.5, orientation2[2]+0.5)]
def optimize(guess):
    return minimize(orientation_cost_function, guess,method="SLSQP", bounds=bnds, options={'ftol': 1e-2,'eps':1e-1, 'finite_diff_rel_step' : 1}, args=(object_name_cad,translation_cad, mesh_scale_cad, camera_intrinsics, max_virtual_depth))
 
 
result = optimize(initial_guesses)


#Create a ThreadPoolExecutor
# with ThreadPoolExecutor(max_workers=20) as executor:
#     # Use the executor to map the optimize function to the initial guesses
#     results = executor.map(optimize, initial_guesses)

# # results now contains the result of the optimization for each initial guess
# result_cost = []
# results_array = []
# for result in results:
#     print("Success:" + str(result.success))
#     print("Result:" + str(result.x))
#     print("Cost:" + str(result.fun))
#     print("Message:" + str(result.message))
#     result_cost.append(result.fun)
#     results_array.append(result.x)

########## plot cost function

from scipy.interpolate import griddata
orientation_angles = np.array([element['orientation'] for element in selected_data])

# Create a figure with 3 subplots (3D plots)
fig = plt.figure(figsize=(18, 6))

# Define meshgrid resolution
grid_resolution = 100j
# Subplot 1: Fix the first orientation, vary the other two
ax1 = fig.add_subplot(131, projection='3d')
grid_x1, grid_y1 = np.mgrid[min(orientation_angles[:,1]):max(orientation_angles[:,1]):grid_resolution, min(orientation_angles[:,2]):max(orientation_angles[:,2]):grid_resolution]

unique_elements, counts = np.unique(orientation_angles[:,0], return_counts=True)
element_index = np.where(unique_elements == orientation2[0])[0][0]
element = unique_elements[element_index]
indices = np.array(np.where(orientation_angles[:,0] == element)).T
grid_z1 = griddata(orientation_angles[indices,1:3].reshape(counts[element_index],2), costs[indices], (grid_x1, grid_y1), method='cubic')
grid_z1 = np.squeeze(grid_z1[:, :, 0])

# grid_z1 = griddata(orientation_angles[:,1:3], costs, (grid_x1, grid_y1), method='cubic')
ax1.plot_surface(grid_x1, grid_y1, grid_z1, cmap='viridis', edgecolor='none')
ax1.set_title('Fixed Orientation 1')
ax1.set_xlabel('Orientation 2')
ax1.set_ylabel('Orientation 3')
ax1.set_zlabel('Cost')

# Subplot 2: Fix the second orientation, vary the others
ax2 = fig.add_subplot(132, projection='3d')
grid_x2, grid_y2 = np.mgrid[min(orientation_angles[:,0]):max(orientation_angles[:,0]):grid_resolution, min(orientation_angles[:,2]):max(orientation_angles[:,2]):grid_resolution]
unique_elements, counts = np.unique(orientation_angles[:,1], return_counts=True)
element_index = np.where(unique_elements == orientation2[1])[0][0]
element = unique_elements[element_index]
indices = np.array(np.where(orientation_angles[:,1] == element)).T
angles_2 = orientation_angles[indices, :].reshape(counts[element_index], 3)
grid_z2 = griddata(angles_2[:,[0,2]], costs[indices], (grid_x2, grid_y2), method='cubic')
grid_z2 = np.squeeze(grid_z2[:, :, 0])

#grid_z2 = griddata(orientation_angles[:,[0,2]], costs, (grid_x2, grid_y2), method='cubic')
ax2.plot_surface(grid_x2, grid_y2, grid_z2, cmap='viridis', edgecolor='none')
ax2.set_title('Fixed Orientation 2')
ax2.set_xlabel('Orientation 1')
ax2.set_ylabel('Orientation 3')
ax2.set_zlabel('Cost')

# Subplot 3: Fix the third orientation, vary the others
ax3 = fig.add_subplot(133, projection='3d')
grid_x3, grid_y3 = np.mgrid[min(orientation_angles[:,0]):max(orientation_angles[:,0]):grid_resolution, min(orientation_angles[:,1]):max(orientation_angles[:,1]):grid_resolution]
unique_elements, counts = np.unique(orientation_angles[:,2], return_counts=True)
element_index = np.where(unique_elements == orientation2[2])[0][0]
element = unique_elements[element_index]
indices = np.array(np.where(orientation_angles[:,2] == element)).T
grid_z3 = griddata(orientation_angles[indices,0:2].reshape(counts[element_index],2), costs[indices], (grid_x3, grid_y3), method='cubic')
grid_z3 = np.squeeze(grid_z3[:, :, 0])

#grid_z3 = griddata(orientation_angles[:,0:2], costs, (grid_x3, grid_y3), method='cubic')
ax3.plot_surface(grid_x3, grid_y3, grid_z3, cmap='viridis', edgecolor='none')
ax3.set_title('Fixed Orientation 3')
ax3.set_xlabel('Orientation 1')
ax3.set_ylabel('Orientation 2')
ax3.set_zlabel('Cost')

# in each subplot, plot the initial guess and the real solution
ax1.scatter(orientation2[1], orientation2[2], cost_min, color='red', s=100)
ax2.scatter(orientation2[0], orientation2[2], cost_min, color='red', s=100)
ax3.scatter(orientation2[0], orientation2[1], cost_min, color='red', s=100)

step_size = 0.1
plot_array = np.arange(0, 1, step_size)
for i in plot_array:
    ax1.scatter(orientation_real[1], orientation_real[2], i, color='blue', s=10)
    ax2.scatter(orientation_real[0], orientation_real[2], i, color='blue', s=10)
    ax3.scatter(orientation_real[0], orientation_real[1], i, color='blue', s=10)
    ax1.scatter(result.x[1], result.x[2], i, color='black', s=10)
    ax2.scatter(result.x[0], result.x[2], i, color='black', s=10)
    ax3.scatter(result.x[0], result.x[1], i, color='black', s=10)


# plot indipendent varuiables and cost 
for i,element in enumerate(indipendent_vars):
    ax1.scatter(element[1], element[2], indipendent_cost[i], color='green', s=50)
    ax2.scatter(element[0], element[2], indipendent_cost[i], color='green', s=50)
    ax3.scatter(element[0], element[1], indipendent_cost[i], color='green', s=50)

ax1.scatter(result.x[1], result.x[2], result.fun, color='black', s=100) 
ax2.scatter(result.x[0], result.x[2], result.fun, color='black', s=100) 
ax3.scatter(result.x[0], result.x[1], result.fun, color='black', s=100) 





plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.savefig('cost_function_3d_continuous_all.png')
plt.show()





###############################

################### RESULT ELABORATION ####################################################################
    
orientation2 = result.x #results_array[np.argmin(result_cost)]


rtheta = geometric_helper.axis_angle_viewpoint(orientation2[0],orientation2[1],orientation2[2])
axis, angle = geometric_helper.axis_angle_from_vector(rtheta)
quaternion2 = geometric_helper.axis_angle_to_quaternion(axis, angle)


# place it in the virtual world
depth_map2, object_pixels2 = image_helper.generate_depth_map(object_name_cad,translation_cad, quaternion2, mesh_scale_cad, camera_intrinsics, max_virtual_depth)
obj_depth_image2 = image_helper.crop_object_image(depth_map2,object_pixels2)
obj_depth_image2_normalized = image_helper.normalize_depth_map(obj_depth_image2)    
# cv2.imshow("depth_map2", depth_map2)


# Retrieve objects point clouds and resample them to get two ordered point clouds
resized_image_real_object, resized_image_cad_model = image_helper.resize_images_to_same_size(obj_depth_image, obj_depth_image2)
# cv2.imshow("resized_image_real_object", image_helper.normalize_depth_map(resized_image_real_object))
# cv2.imshow("resized_image_cad_model", image_helper.normalize_depth_map(resized_image_cad_model))

res_height,res_width = resized_image_real_object.shape
resampled_depth_map_real = image_helper.resample_depth_map(depth_map, object_pixels,res_width,res_height)
resampled_depth_map_cad = image_helper.resample_depth_map(depth_map2, object_pixels2,res_width,res_height)

point_cloud_real = image_helper.depth_to_pointcloud_fromlist(resampled_depth_map_real,camera_intrinsics)
point_cloud_cad = image_helper.depth_to_pointcloud_fromlist(resampled_depth_map_cad,camera_intrinsics)

point_cloud_real = np.array(point_cloud_real)
point_cloud_cad = np.array(point_cloud_cad)

mask = ~np.isnan(point_cloud_real) & ~np.isnan(point_cloud_cad)
mask_matrix = mask.reshape(res_width*res_height,3)
nan_depth_index = mask_matrix[:,2]

point_cloud_real = point_cloud_real[nan_depth_index]
point_cloud_cad = point_cloud_cad[nan_depth_index]

image_helper.plot_pointcloud(point_cloud_real,"point_cloud_real")
image_helper.plot_pointcloud(point_cloud_cad,"point_cloud_cad")

# now we have two ordered point clouds, we can run Umeyama and retrieve the relative translation, orientation and scale 
R, c, t = geometric_helper.kabsch_umeyama(point_cloud_real, point_cloud_cad)

print("relative translation", t)
print("relative scale", c)
print("relative orientation", R)

point_cloud_cad_transformed = np.array([t + c * R @ b for b in point_cloud_cad])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = [point[0] for point in point_cloud_real]
ys = [point[1] for point in point_cloud_real]
zs = [point[2] for point in point_cloud_real]
ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
xs = [point[0] for point in point_cloud_cad_transformed]
ys = [point[1] for point in point_cloud_cad_transformed]
zs = [point[2] for point in point_cloud_cad_transformed]
ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
xs = [point[0] for point in point_cloud_cad]
ys = [point[1] for point in point_cloud_cad]
zs = [point[2] for point in point_cloud_cad]
ax.scatter(xs, ys, zs, c='r'), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
xs = [translation_cad[0]]
ys = [translation_cad[1]]
zs = [translation_cad[2]]
ax.scatter(xs, ys, zs, c='g'), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
xs = [translation_real[0]]
ys = [translation_real[1]]
zs = [translation_real[2]]
ax.scatter(xs, ys, zs, c='b'), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
# plt.show()
plt.savefig("Result" + '.png')

#####################################################
# Compute real object estimated pose wrt camera frame 

R2 = geometric_helper.quaternion_to_rotation_matrix(quaternion2)

estimated_p_real = t + c * R @ np.array(translation_cad).T
estimated_R_real = geometric_helper.normalize_rotation_matrix(c * R @ R2)  #???
estimated_scale_real = mesh_scale_cad*c

print("real object position", estimated_p_real)
print("real object orientation", estimated_R_real) 
print("real object scale", estimated_scale_real)

cv2.waitKey(0)
cv2.destroyAllWindows()
nvisii.deinitialize()