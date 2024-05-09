import numpy as np
from plyfile import PlyData
import cv2
import matplotlib.pyplot as plt
import math

def generate_depth_map(vertices, intrinsic_matrix, extrinsic_matrix, image_dimensions):
    # Convert vertices to homogeneous coordinates (add 1)
    homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))

    # Apply extrinsic transformation
    transformed_vertices = np.dot(homogeneous_vertices, extrinsic_matrix.T)

    # Project vertices into the camera
    projected_vertices = np.dot(transformed_vertices, intrinsic_matrix.T)

    # Normalize homogeneous coordinates
    projected_vertices[:, 0] /= projected_vertices[:, 2]
    projected_vertices[:, 1] /= projected_vertices[:, 2]

    # Create the depth map
    depth_map = np.ones(image_dimensions)* np.nan
    object_pixels = []

    # Draw projected points on the depth map
    for point in projected_vertices:
        x, y, z = point.astype(float)
        if 0 <= x < image_dimensions[1] and 0 <= y < image_dimensions[0]:
            xr = math.floor(x)
            yr = math.floor(y)
            if math.isnan(depth_map[yr,xr]):          
                depth_map[yr, xr] = z  # Depth Z
                object_pixels.append((xr, yr))  # Add the pixel to the list of object pixels
            else:        
            	# the pixel is already occupied => take the nearest one
            	depth_map[yr,xr] = min(z,depth_map[yr,xr])
            	
            	

    return depth_map, object_pixels
            
    
def remove_occluded_points(depth_map,object_pixels):
    # Define neighborhood size
    neighborhood_size = 10
    # Get dimensions of the depth map
    height, width = depth_map.shape

    # Create an empty array to store the result
    result_map = np.zeros((height, width))
    

    # Iterate over each pixel in the depth map
    for x, y in object_pixels:
            # Get the depth value of the current pixel
            depth_value = depth_map[y, x]
            

            # Check if the current pixel is at the border where the neighborhood is not fully available
            if y - neighborhood_size >= 0 and y + neighborhood_size < height and \
                    x - neighborhood_size >= 0 and x + neighborhood_size < width:
                # Extract the neighborhood around the current pixel
                neighborhood = depth_map[y - neighborhood_size:y + neighborhood_size + 1,
                                          x - neighborhood_size:x + neighborhood_size + 1]
                                     

                # Compute the mean depth value of the neighborhood
                mean_depth = np.nanmean(neighborhood)

                # Check if the depth value of the current pixel is less than the mean depth of the neighborhood
                if depth_value <= mean_depth:
                    result_map[y, x] = depth_value

    return result_map

def read_ply_file(file_name):
    try:
        # Read the PLY file
        with open(file_name, 'rb') as f:
            ply_data = PlyData.read(f)

        # Extract necessary information from the PLY file
        vertices = np.vstack([ply_data['vertex']['x'],
                              ply_data['vertex']['y'],
                              ply_data['vertex']['z']]).T

        # Example of extracting intrinsic and extrinsic parameters
        intrinsic_matrix = np.array([[focal_length_x, 0, principal_point_x],
                                      [0, focal_length_y, principal_point_y],
                                      [0, 0, 1]])

        extrinsic_matrix = np.array([[rotation_00, rotation_01, rotation_02, translation_x],
                                      [rotation_10, rotation_11, rotation_12, translation_y],
                                      [rotation_20, rotation_21, rotation_22, translation_z]])

        image_dimensions = (image_height, image_width)

        return vertices, intrinsic_matrix, extrinsic_matrix, image_dimensions

    except FileNotFoundError:
        print("The specified file was not found.")
        return None, None, None, None
    except Exception as e:
        print("An error occurred while reading the file:", e)
        return None, None, None, None
        
        


# Example initialization of intrinsic and extrinsic parameters
focal_length_x = 1000  # Focal length in pixels (along X-axis)
focal_length_y = 1000  # Focal length in pixels (along Y-axis)
principal_point_x = 320  # Principal point offset along X-axis (in pixels)
principal_point_y = 240  # Principal point offset along Y-axis (in pixels)

# Example initialization of extrinsic parameters (rotation and translation)
rotation_00 = 1.0  # Rotation element [0, 0] of the rotation matrix
rotation_01 = 0.0  # Rotation element [0, 1] of the rotation matrix
rotation_02 = 0.0  # Rotation element [0, 2] of the rotation matrix
rotation_10 = 0.0  # Rotation element [1, 0] of the rotation matrix
rotation_11 = 1.0  # Rotation element [1, 1] of the rotation matrix
rotation_12 = 0.0  # Rotation element [1, 2] of the rotation matrix
rotation_20 = 0.0  # Rotation element [2, 0] of the rotation matrix
rotation_21 = 0.0  # Rotation element [2, 1] of the rotation matrix
rotation_22 = 1.0  # Rotation element [2, 2] of the rotation matrix

translation_x = 0.0  # Translation along X-axis (in the camera coordinate system)
translation_y = 0.0  # Translation along Y-axis (in the camera coordinate system)
translation_z = 4  # Translation along Z-axis (in the camera coordinate system)

image_height = 640
image_width = 480

# Example usage
file_name_ply = "cube_resampled_rot.ply"  # Replace with the path to your PLY file
vertices, K, Rt, image_dimensions = read_ply_file(file_name_ply)



if vertices is not None:
    # Generate the depth map
    depth_map, object_pixels = generate_depth_map(vertices, K, Rt, image_dimensions)
    
    # Remove occluded points
    visible_vertices = remove_occluded_points(depth_map,object_pixels) 
   
    # Display the depth map with only visible vertices
    cv2.imshow("Depth Map", depth_map) 
    cv2.imshow("Depth Map ref", visible_vertices)

    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

