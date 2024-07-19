import numpy as np
import cv2
import geometric_helper as geometric_helper
import image_helper as image_helper
import nvisii_helper as nvisii_helper

# File to save the viewpoint data
viewpoint_filename = "hammer_viewpoints_20aa.pkl"

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
file_name = "cad_models/hammer.obj"  
mesh_scale = 1 #0.01 banana


max_virtual_depth = 5 #[m]


# Pose object
translation = np.array([0,0,1]) # position of the object in meters wrt camera
euler_angles = [0,0,0] # radians - roll pitch and yaw
quaternion_real = geometric_helper.euler_to_quaternion(euler_angles)

# initialize nvisii
interactive = False
nvisii_helper.initialize_nvisii(interactive, camera_intrinsics,object_name, file_name)


# Generate the real depth map
depth_map, object_pixels = image_helper.generate_depth_map(object_name,translation, quaternion_real, mesh_scale, camera_intrinsics, max_virtual_depth) # The first time call it two times due to nvisii bug


# creat a three numpy array that range from -pi to pi with specified step_size
step_size = 20*np.pi/180
phi_array = np.arange(0+step_size,np.pi-step_size, step_size)
theta_array = np.arange(0+step_size, np.pi-step_size, step_size)
psi_array = np.arange(0+step_size, 2*np.pi-step_size,  step_size)

# iterate over the three arrays and generate for each the obj_depth_image
number_of_iteration = 0
number_of_iterations = len(theta_array) * len(phi_array) * len(psi_array)
data = []

for phi in phi_array:
    for theta in theta_array:
        for psi in psi_array:
            #quaternion = geometric_helper.euler_to_quaternion([phi,theta,psi])
            axis, angle = geometric_helper.axis_angle_from_vector(geometric_helper.axis_angle_viewpoint(phi,theta,psi))

            
            quaternion = geometric_helper.axis_angle_to_quaternion(axis, angle)
            
            # Generate the depth map
            depth_map, object_pixels = image_helper.generate_depth_map(object_name,translation, quaternion, mesh_scale, camera_intrinsics, max_virtual_depth)
            # crop object image
            obj_depth_image = image_helper.crop_object_image(depth_map,object_pixels)
            # normalize object depth map
            obj_depth_image_normalized = image_helper.normalize_depth_map(obj_depth_image)
            # compute aspect_ratio
            aspect_ratio = obj_depth_image_normalized.shape[1] / obj_depth_image_normalized.shape[0]
            
            print("phi", phi, "theta", theta, "psi", psi, "aspect_ratio", aspect_ratio)
            print("image size [byte]", len(object_pixels)*4)
            print("iteration", number_of_iteration, "out of", number_of_iterations)
            
            
            # Store the data for this iteration
            data.append({
                'orientation': [phi, theta, psi],
                'depth_map': obj_depth_image_normalized,
                'aspect_ratio': aspect_ratio
            })
            
            number_of_iteration = number_of_iteration + 1
            
            cv2.imshow("Object image", obj_depth_image_normalized)
            cv2.waitKey(10)
            
# Save the data to a file
import pickle
with open(viewpoint_filename, 'wb') as f:
    pickle.dump(data, f)
    
    
cv2.destroyAllWindows()
nvisii_helper.deinitialize_nvisii()
