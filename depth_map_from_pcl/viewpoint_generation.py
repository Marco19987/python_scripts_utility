import numpy as np
import cv2
import nvisii
import geometric_helper as geometric_helper
import image_helper as image_helper
import nvisii_utils as nvisii_utils

# Initialization of intrinsic and extrinsic parameters
focal_length_x = 610.0  # Focal length in pixels (along X-axis)
focal_length_y = 610.0  # Focal length in pixels (along Y-axis)
principal_point_x = 317.0  # Principal point offset along X-axis (in pixels)
principal_point_y = 238.0  # Principal point offset along Y-axis (in pixels)
image_height = 480
image_width = 640
camera_intrinsics = [focal_length_x,focal_length_y,principal_point_x,principal_point_y,image_width,image_height]


# Load file real object
object_name = "banana"
file_name = "cad_models/connectorMO6.obj"  
mesh_scale = 0.01 #0.01 banana


max_virtual_depth = 5 #[m]


# Pose object
translation = np.array([0,0,0.2]) # position of the object in meters wrt camera
euler_angles = [0,0,0] # radians - roll pitch and yaw
quaternion_real = geometric_helper.euler_to_quaternion(euler_angles)

# initialize nvisii
interactive = False
nvisii_utils.initialize_nvisii(interactive, camera_intrinsics,object_name, file_name)


# Generate the real depth map
depth_map, object_pixels = image_helper.generate_depth_map(object_name,translation, quaternion_real, mesh_scale, camera_intrinsics, max_virtual_depth) # The first time call it two times due to nvisii bug


# creat a three numpy array that range from -pi to pi with specified step_size
step_size = 45*np.pi/180
theta_array = np.arange(-np.pi, np.pi, step_size)
phi_array = np.arange(-np.pi, np.pi, step_size)
psi_array = np.arange(-np.pi, np.pi, step_size)

# iterate over the three arrays and generate for each the obj_depth_image
number_of_iteration = 0
number_of_iterations = len(theta_array) * len(phi_array) * len(psi_array)
data = []
for theta in theta_array:
    for phi in phi_array:
        for psi in psi_array:
            quaternion = geometric_helper.euler_to_quaternion([theta,phi,psi])
            # Generate the depth map
            depth_map, object_pixels = image_helper.generate_depth_map(object_name,translation, quaternion, mesh_scale, camera_intrinsics, max_virtual_depth)
            # crop object image
            obj_depth_image = image_helper.crop_object_image(depth_map,object_pixels)
            # normalize object depth map
            obj_depth_image_normalized = geometric_helper.normalize_depth_map(obj_depth_image)
            # compute aspect_ratio
            aspect_ratio = obj_depth_image_normalized.shape[1] / obj_depth_image_normalized.shape[0]

            # flipud the depth map
            # obj_depth_image_normalized_flipud = np.flipud(obj_depth_image_normalized)
            # obj_depth_image_normalized_fliplr = np.fliplr(obj_depth_image_normalized)

            
            print("theta", theta, "phi", phi, "psi", psi, "aspect_ratio", aspect_ratio)
            print("image size [byte]", len(object_pixels)*4)
            print("iteration", number_of_iteration, "out of", number_of_iterations)
            
            
            # Store the data for this iteration
            data.append({
                'euler_angles': [theta, phi, psi],
                'depth_map': obj_depth_image_normalized,
                'aspect_ratio': aspect_ratio
            })
            # cv2.imshow("Object image flipud",obj_depth_image_normalized_flipud)
            # cv2.imshow("Object image fliplr",obj_depth_image_normalized_fliplr)
            #cv2.imshow("Object image", depth_map)
            #cv2.waitKey(0)
            
            number_of_iteration = number_of_iteration + 1
            
            
# Save the data to a file
import pickle
with open('connectorMO6_viewpoints_45.pkl', 'wb') as f:
    pickle.dump(data, f)
    
    
cv2.destroyAllWindows()
nvisii_utils.deinitialize_nvisii()
