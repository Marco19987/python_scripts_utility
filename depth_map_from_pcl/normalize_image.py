import numpy as np
from scipy.optimize import minimize
from plyfile import PlyData
import cv2
import matplotlib.pyplot as plt
import math
import trimesh

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
        
        
def crop_object_image(depth_map,object_pixels):
    
    pixel_x , pixel_y = map(list, zip(*object_pixels))
    max_x = max(pixel_x)
    max_y = max(pixel_y)
    min_x = min(pixel_x)
    min_y = min(pixel_y)
        
    image_dimensions = [max_y-min_y+1,max_x-min_x+1]
    print(image_dimensions)
    # Create the depth map
    obj_image = np.ones(image_dimensions)*np.nan


    pixel_x = list(range(min_x,max_x))    
    pixel_y = list(range(min_y,max_y))
    
    for x in pixel_x:
        for y in pixel_y:
            #print(min_y -1+ y,min_x -1 + x)
            obj_image[y - min_y,x - min_x] = depth_map[y,x] 

    
    return obj_image


def read_obj_file(file_name):
    try:
        # Read the OBJ file
        mesh = trimesh.load(file_name)
    
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        return vertices, faces

    except FileNotFoundError:
        print("The specified file was not found.")
        return None, None, None, None
    except Exception as e:
        print("An error occurred while reading the file:", e)
        return None, None, None, None
    
def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion into a 3x3 rotation matrix.
    
    Parameters:
    q (tuple or list): A quaternion represented as (w, x, y, z)
    
    Returns:
    np.ndarray: A 3x3 rotation matrix
    """
    w, x, y, z = q
    
    # Compute the elements of the rotation matrix
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    
    return R


def normalize_quaternion(q):
    """
    Normalizes a quaternion.
    
    Parameters:
    q (tuple or list): A quaternion represented as (w, x, y, z)
    
    Returns:
    tuple: The normalized quaternion
    """
    w, x, y, z = q
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    return w / norm, x / norm, y / norm, z / norm

def normalize_depth_map(depth_map):
    
    max_value = np.nanmax(depth_map)
    min_value = np.nanmin(depth_map)
    
    range_value = max_value - min_value
    
    # Normalize the matrix
    normalized_depth_map = (depth_map - min_value)/ range_value
    
    # row,col = normalized_depth_map.shape
    # a = 0
    # for r in range(row):
    #     for c in range(col):
    #         if(math.isnan(normalized_depth_map[r][c])):
    #             a = a
    #         else:
    #             a = a + normalized_depth_map[r][c]
    #             #print(normalized_depth_map[r][c])


    return normalized_depth_map

def resize_images_to_same_size(image1_array, image2_array):
    """
    Resize two images (given as numpy arrays) to the same size, based on the smaller dimensions of the two.
    
    Parameters:
    image1_array (np.array): The first image as a numpy array.
    image2_array (np.array): The second image as a numpy array.
    
    Returns:
    np.array: The resized first image.
    np.array: The resized second image.
    """
    # Get the dimensions of the images
    height_1, width_1 = image1_array.shape
    height_2, width_2 = image2_array.shape
    
    dimensions = [width_1, height_1, width_2, height_2]
    maximum = max(dimensions)
    index_max = dimensions.index(maximum)
    print(dimensions)
    
    
    # Determine the new size based on the smaller dimensions
    if index_max < 2:
        # Resize first image
        if index_max == 0:
            # Width
            new_height = height_2
            new_width = int((height_2 / height_1) * width_1)
            # resized_image = np.ones((new_height,new_width))*np.nan
            # disp = 0
            # if height_1 <= height_2:
            #     disp = int(width_2/2)
            # for y in range(disp, disp+width_2):
            #     for x in range(height_2):                    
            #         resized_image[x][y] = image2_array[x][y-disp]
        else:
            # Height
            new_width = width_2
            new_height = int((width_2 / width_1) * height_1)
            # resized_image = np.ones((new_height,new_width))*np.nan
            # disp = 0
            # if width_1 <= width_2:
            #     disp = int(height_2/2)
            # for x in range(disp, disp+height_2):
            #     for y in range(width_2):                    
            #         resized_image[x][y] = image2_array[x-disp][y]
        
        resized_image_1 = cv2.resize(image1_array, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        not_resized_image = image2_array
    else:
        # Resize second image
        if index_max == 2:
            # Width
            new_height = height_1
            new_width = int((height_1 / height_2) * width_2)
            # print(new_width,new_height)
            # resized_image = np.ones((new_height,new_width))*np.nan
            # disp = 0
            # if height_2 <= height_1:
            #     disp = int(width_1/2)
            # for y in range(disp, disp+width_1):
            #     for x in range(height_1):                 
            #         resized_image[x][y] = image1_array[x][y-disp]


        else:
            # Height
            new_width = width_1
            new_height = int((width_1 / width_2) * height_2)
            # resized_image = np.ones((new_height,new_width))*np.nan
            # disp = 0
            # # if width_2 <= width_1:
            #     disp = int(height_1/2)
            # for x in range(disp, disp+height_1):
            #     for y in range(width_1):                    
            #         resized_image[x][y] = image1_array[x-disp][y]
                    
        resized_image_1 = cv2.resize(image2_array, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        not_resized_image = image1_array
        
    # resize the images to have the same dimensions
    res_height_1, res_width_1 = resized_image_1.shape
    res_height_2, res_width_2 = not_resized_image.shape
        
    if res_height_1 == res_height_2 and res_width_1 == res_width_2:
        # the images have the same size
        resized_image_2 = not_resized_image
    else:
        if res_height_1 == res_height_2:
            # scale width
            tmp_image = [resized_image_1,not_resized_image]
            res_dim = [res_width_1,res_width_2]
            min_width_index = res_dim.index(min(res_dim))
            max_width_index = res_dim.index(max(res_dim))
            resized_image = np.ones((res_height_1,res_dim[max_width_index]))*np.nan
            print("new image dime", res_height_1,res_dim[max_width_index])
            disp = int((res_dim[max_width_index]-res_dim[min_width_index])/2)
            for y in range(disp, disp+res_dim[min_width_index]):
                 for x in range(res_height_1):
                     resized_image[x][y] = tmp_image[min_width_index][x][y-disp]
            resized_image_1 = tmp_image[max_width_index]
            resized_image_2 = resized_image
        else:
            # scale height
            tmp_image = [resized_image_1,not_resized_image]
            res_dim = [res_height_1,res_height_2]
            min_height_index = res_dim.index(min(res_dim))
            max_height_index = res_dim.index(max(res_dim))
            resized_image = np.ones((res_dim[max_height_index],res_width_1))*np.nan
            disp = int((res_dim[max_height_index]-res_dim[min_height_index])/2)
            for x in range(disp, disp+res_dim[min_height_index]):
                for y in range(res_width_1):                    
                    resized_image[x][y] = tmp_image[min_height_index][x-disp][y]
            resized_image_1 = tmp_image[max_height_index]
            resized_image_2 = resized_image

    return resized_image_1, resized_image_2

def getRtmatrix(translation, quaternion):
    """
    Given a quaternion and a translation vector, returns the Rt matrix.
    
    Parameters:
    quaternion (tuple or list): A quaternion represented as (w, x, y, z).
    translation (tuple or list): A translation vector represented as (tx, ty, tz).
    
    Returns:
    np.ndarray: A 3x4 transformation matrix combining rotation and translation.
    """
    # Normalize the quaternion
    w, x, y, z = normalize_quaternion(quaternion)
    
    # Compute the rotation matrix from the quaternion
    R = quaternion_to_rotation_matrix((w, x, y, z))
    
    # Combine the rotation matrix and translation vector to form the Rt matrix
    tx, ty, tz = translation
    Rt = np.array([
        [R[0, 0], R[0, 1], R[0, 2], tx],
        [R[1, 0], R[1, 1], R[1, 2], ty],
        [R[2, 0], R[2, 1], R[2, 2], tz]
    ])
    
    return Rt

def orientation_cost_function(quaternion):
    
    translation2 = [0,0,0.5]
    quaternion2 = quaternion
    Rt2 = getRtmatrix(translation2,quaternion2)
    depth_map2, object_pixels2 = generate_depth_map(vertices, K, Rt2, image_dimensions)
    obj_depth_image2 = crop_object_image(depth_map2,object_pixels2)
    obj_depth_image2 = normalize_depth_map(obj_depth_image2)    
    resized_image1_array, resized_image2_array = resize_images_to_same_size(obj_depth_image2, obj_depth_image)
    
    cv2.imshow("resized_image1 ", resized_image1_array)
    cv2.imshow("resized_image2 ", resized_image2_array)
    cv2.waitKey(0)
    
    w1,h1 = obj_depth_image.shape
    w2,h2 = obj_depth_image2.shape
    
    aspect_ratio_1 = w1/h1
    aspect_ratio_2 = w2/h2

    

    
    # Compute the cost as the sum of squared differences of non-NaN pixels
    mask = ~np.isnan(resized_image1_array) & ~np.isnan(resized_image2_array)
    cost = np.nansum((resized_image1_array[mask] - resized_image2_array[mask]) ** 2)
    cost = cost/(resized_image1_array[mask].size)
    cost = cost + (aspect_ratio_1-aspect_ratio_2)**2
    print("quaternion", quaternion)
    print("cost value", cost)

    return cost

def unit_quaternion_constraint(quaternion):
    return np.sum(np.square(quaternion)) - 1

# Example initialization of intrinsic and extrinsic parameters
focal_length_x = 610  # Focal length in pixels (along X-axis)
focal_length_y = 610  # Focal length in pixels (along Y-axis)
principal_point_x = 317  # Principal point offset along X-axis (in pixels)
principal_point_y = 238  # Principal point offset along Y-axis (in pixels)

image_height = 640
image_width = 480
image_dimensions = (image_height, image_width)

# Example of extracting intrinsic and extrinsic parameters
K = np.array([[focal_length_x, 0, principal_point_x],
                                [0, focal_length_y, principal_point_y],
                                [0, 0, 1]])


# Example initialization of extrinsic parameters (rotation and translation)
quaternion = [0,0,0,1]  
translation = np.array([0,0,0.5])


Rt = getRtmatrix(translation, quaternion)


# Load file
file_name = "Lime.obj"  # Replace with the path to your PLY file
#vertices, K, Rt, image_dimensions = read_ply_file(file_name)
vertices, faces = read_obj_file(file_name)
vertices = vertices*1



if vertices is not None:
    # Generate the depth map
    depth_map, object_pixels = generate_depth_map(vertices, K, Rt, image_dimensions)
    # Remove occluded points
    #visible_vertices = remove_occluded_points(depth_map,object_pixels) 
    
    
    # crop object image
    obj_depth_image = crop_object_image(depth_map,object_pixels)
    
    # normalize object depth map
    obj_depth_image = normalize_depth_map(obj_depth_image)
    
    
    # Ottimizzazione della funzione di costo
    initial_guess = [0, 0, 1, 0]
    constraints = {'type': 'eq', 'fun': unit_quaternion_constraint}
    result = minimize(orientation_cost_function, initial_guess,constraints=constraints)
    #quaternion_optimized = result.x
    
    
    
    # second pose
    translation2 = [0,0,0.5]
    #quaternion2 = quaternion_optimized
    quaternion2 = [0.88127318,0.39834127,0.14929647,0.20589413]
    Rt2 = getRtmatrix(translation2,quaternion2)
    depth_map2, object_pixels2 = generate_depth_map(vertices, K, Rt2, image_dimensions)
    obj_depth_image2 = crop_object_image(depth_map2,object_pixels2)
    obj_depth_image2 = normalize_depth_map(obj_depth_image2)    
    cv2.imshow("Depth Map2", depth_map2)
    cv2.imshow("obj_depth_image2", obj_depth_image2)



    resized_image1_array, resized_image2_array = resize_images_to_same_size(obj_depth_image, obj_depth_image2)

    #print(resized_image1_array.shape)
    #print(resized_image2_array.shape)

    cv2.imshow("Resized Image 1", resized_image1_array)
    cv2.imshow("Resized Image 2", resized_image2_array)
    
    # Display the depth map with only visible vertices
    #cv2.imshow("Depth Map", depth_map)

     
    #cv2.imshow("Depth Map ref", visible_vertices)

    #cv2.imshow("Depth Map object", obj_depth_image)
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

