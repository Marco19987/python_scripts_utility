import tkinter as tk
import numpy as np

def rotation_matrix_to_euler_angles(R):
    # Check that the rotation matrix is valid
    assert np.allclose(np.linalg.det(R), 1.0), "Invalid rotation matrix"

    # Extract the rotation angles
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw
def evaluate_cost(val):
    
    # transform in roll pitch yaw
    rx = slider1.get()
    ry = slider2.get()
    rz = slider3.get()
    print("rx: ", rx)
    print("ry: ", ry)
    print("rz: ", rz)
    
    quaternion = euler_to_quaternion([rx,ry,rz])
    quaternion = normalize_quaternion(quaternion)
    R = quaternion_to_rotation_matrix(quaternion)
    axis, angle = rotation_matrix_to_axis_angle(R)
    orientation = axis*angle
    
    
    cost, depth_map2, res1,res2 = orientation_cost_function(orientation)
    print("cost: ", cost)
    cv2.imshow("depth_map", depth_map)
    cv2.imshow("depth_map2", depth_map2)
    cv2.imshow("Real image", (res1))
    cv2.imshow("Cad model", (res2))
    cv2.waitKey(1)
    
import numpy as np
from scipy.optimize import minimize, dual_annealing,NonlinearConstraint, least_squares,basinhopping,differential_evolution, OptimizeResult
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import nvisii
import time
import matplotlib
from concurrent.futures import ThreadPoolExecutor
import trimesh
from sklearn.neighbors import NearestNeighbors


matplotlib.use('Agg')  # Use a non-interactive backend

def crop_object_image(depth_map,object_pixels):
    
    pixel_h , pixel_w = map(np.array, zip(*object_pixels))
    max_w = pixel_w.max()
    max_h = pixel_h.max()
    min_w = pixel_w.min()
    min_h = pixel_h.min()

    image_dimensions = [max_h-min_h+1, max_w-min_w+1] # height, width

    # Create the depth map
    obj_image = np.full(image_dimensions, np.nan)

    pixel_w = np.arange(min_w, max_w)
    pixel_h = np.arange(min_h, max_h)

    # Use advanced indexing to fill obj_image
    h_indices, w_indices = np.meshgrid(pixel_h - min_h, pixel_w - min_w, indexing='ij')
    obj_image[h_indices, w_indices] = depth_map[pixel_h[:, None], pixel_w]
    
    return obj_image

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
    
    # print("normalization values", max_value, min_value,  range_value, normalized_depth_map)
    
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
    # print(dimensions)
    
    image_resized_index = -1 # index that track the resized image
    switch_images = False # if True switch images in the end
    
    # Determine the new size based on the smaller dimensions
    if index_max < 2:
        # Resize first image
        if index_max == 0:
            # Width
            new_height = height_2
            new_width = int((height_2 / height_1) * width_1)
            
        else:
            # Height
            new_width = width_2
            new_height = int((width_2 / width_1) * height_1)
            
        
        resized_image_1 = cv2.resize(image1_array,(new_width, new_height), interpolation=cv2.INTER_NEAREST)
        not_resized_image = image2_array
        image_resized_index = 0 # image 1 resized

    else:
        # Resize second image
        if index_max == 2:
            # Width
            new_height = height_1
            new_width = int((height_1 / height_2) * width_2)
        else:
            # Height
            new_width = width_1
            new_height = int((width_1 / width_2) * height_2)
            
                    
        resized_image_1 = cv2.resize(image2_array,(new_width,new_height), fx=focal_length_x, fy=focal_length_y, interpolation=cv2.INTER_NEAREST)
        not_resized_image = image1_array
        image_resized_index = 1 # image 2 resized

        
    # resize the images to have the same dimensions
    res_height_1, res_width_1 = resized_image_1.shape
    res_height_2, res_width_2 = not_resized_image.shape

        
    if res_height_1 == res_height_2 and res_width_1 == res_width_2:
        # the images have the same size
        resized_image_1 = resized_image_1
        resized_image_2 = not_resized_image
        if image_resized_index==1:
            switch_images = True
        
    else:
        if res_height_1 == res_height_2:
            # scale width
            tmp_image = [resized_image_1,not_resized_image]
            res_dim = [res_width_1,res_width_2]
            min_width_index = res_dim.index(min(res_dim))
            max_width_index = res_dim.index(max(res_dim))
            resized_image = np.ones((res_height_1,res_dim[max_width_index]))*np.nan
            disp = int((res_dim[max_width_index]-res_dim[min_width_index])/2)
            for w in range(disp, disp+res_dim[min_width_index]):
                 for h in range(res_height_1):
                     resized_image[h][w] = tmp_image[min_width_index][h][w-disp]
            resized_image_1 = tmp_image[max_width_index]
            resized_image_2 = resized_image
            if min_width_index == image_resized_index:
                switch_images = True
          
        else:
            # scale height
            tmp_image = [resized_image_1,not_resized_image]
            res_dim = [res_height_1,res_height_2]
            min_height_index = res_dim.index(min(res_dim))
            max_height_index = res_dim.index(max(res_dim))
            resized_image = np.ones((res_dim[max_height_index],res_width_1))*np.nan
            disp = int((res_dim[max_height_index]-res_dim[min_height_index])/2)
            for h in range(disp, disp+res_dim[min_height_index]):
                for w in range(res_width_1):                    
                    resized_image[h][w] = tmp_image[min_height_index][h-disp][w]
            resized_image_1 = tmp_image[max_height_index]
            resized_image_2 = resized_image
            if min_height_index == image_resized_index:
                switch_images = True
                
    if switch_images:
        tmp = resized_image_2
        resized_image_2 = resized_image_1
        resized_image_1 = tmp
            

    return resized_image_1, resized_image_2


# def resize_images_to_same_size(image1_array, image2_array):
#     """
#     Resize two images (given as numpy arrays) to the same size, based on the smaller dimensions of the two.
    
#     Parameters:
#     image1_array (np.array): The first image as a numpy array.
#     image2_array (np.array): The second image as a numpy array.
    
#     Returns:
#     np.array: The resized first image.
#     np.array: The resized second image.
#     """
#     # Get the dimensions of the images
#     height_1, width_1 = image1_array.shape[:2]
#     height_2, width_2 = image2_array.shape[:2]
    
#     # Determine the new size based on the smaller dimensions
#     if height_1 * width_1 > height_2 * width_2:
#         new_height = height_2
#         new_width = int((height_2 / height_1) * width_1)
#         resized_image_1 = cv2.resize(image1_array, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
#         resized_image_2 = image2_array
#     else:
#         new_height = height_1
#         new_width = int((height_1 / height_2) * width_2)
#         resized_image_2 = cv2.resize(image2_array, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
#         resized_image_1 = image1_array

#     # resize the images to have the same dimensions
#     max_height = max(resized_image_1.shape[0], resized_image_2.shape[0])
#     max_width = max(resized_image_1.shape[1], resized_image_2.shape[1])

#     final_image_1 = np.full((max_height, max_width), np.nan)
#     final_image_2 = np.full((max_height, max_width), np.nan)

#     final_image_1[:resized_image_1.shape[0], :resized_image_1.shape[1]] = resized_image_1
#     final_image_2[:resized_image_2.shape[0], :resized_image_2.shape[1]] = resized_image_2

#     return final_image_1, final_image_2


# def orientation_cost_function(orientation):
#     print(orientation)
#     #translation_cad = [-0.2,-0.1,0.5]
        
#     if np.linalg.norm(orientation) != 0:
#         theta = np.linalg.norm(orientation)
#         axis = orientation/theta
#     else:
#         theta = 2*np.pi #identity
#         axis = [0,0,1]
        
#     # print("axis", axis)
#     # print("theta", theta)        
#     orientation = np.concatenate((axis, [theta]))
    

    
#     # quaternion2 = euler_to_quaternion(orientation)
#     # quaternion2 = orientation
#     quaternion2 = axis_angle_to_quaternion(orientation[0:3],orientation[3])
    
#     quaternion2 = normalize_quaternion(quaternion2)
#     depth_map2, object_pixels2 = generate_depth_map(object_name,translation_cad, quaternion2)
   
#     # # crop object image
#     obj_depth_image2 = crop_object_image(depth_map2,object_pixels2)
    
#     # # normalize object depth map
#     obj_depth_image2_normalized = normalize_depth_map(obj_depth_image2)
    
#     resized_image1_array, resized_image2_array = resize_images_to_same_size(obj_depth_image_normalized, obj_depth_image2_normalized)
        
#     h1, w1 = obj_depth_image_normalized.shape
#     h2, w2 = obj_depth_image2_normalized.shape
    
#     aspect_ratio_1 = w1/h1
#     aspect_ratio_2 = w2/h2

    
#     h_image,w_image = resized_image1_array.shape
#     cost = 0
#     for w in range(w_image):
#         for h in range(h_image):
#             d1 = resized_image1_array[h][w]
#             d2 = resized_image2_array[h][w]
                        
#             if math.isnan(d1) and math.isnan(d2):  
#                 cost = cost + 0 # is good
#             else:
#                 if math.isnan(d2) or math.isnan(d1):
#                     cost = cost + 1
#                 else:
#                     # print("depth_values", d1,d2)
#                     cost = cost + 1*pow((d1-d2),2)
                    

#     #cost = cost/(w*h)
#     cost = cost/(h_image*w_image)
#     # cost = cost/(h1*w1)
#     cost = cost + 0*(aspect_ratio_1-aspect_ratio_2)**2 
    

#     # # Replace NaN values in the depth arrays
#     # res_image1 = np.nan_to_num(resized_image1_array, nan=10.0)
#     # res_image2 = np.nan_to_num(resized_image2_array, nan=10.0)


#     # # Generate edge maps for the real and virtual depth arrays
#     # edge_map1 = cv2.Sobel(res_image1, cv2.CV_64F, 1, 1, ksize=1)
#     # edge_map2 = cv2.Sobel(res_image2, cv2.CV_64F, 1, 1, ksize=1)

#     # cv2.imshow("edge_map1", edge_map1)
#     # cv2.imshow("edge_map2", edge_map2)
#     # edge_difference = edge_map1 - edge_map2
#     # edge_difference = np.nan_to_num(edge_difference, nan=1.0)
#     # delta = 1.0
#     # edge_huber_loss = np.where(np.abs(edge_difference) < delta, 0.5 * np.square(edge_difference), delta * (np.abs(edge_difference) - 0.5 * delta))

#     # cost = cost + np.nansum(edge_huber_loss)/(w_image*h_image)
#     # print("orientation", orientation)
#     # print("cost value", cost)
#     # print("aspect ratios", aspect_ratio_1, aspect_ratio_2)
    
#     # cv2.imshow("depth_map2", depth_map2)
#     # cv2.imshow("Real image", resized_image1_array)
#     # cv2.imshow("Cad model", resized_image2_array)
#     # cv2.waitKey(0)

#     return cost


def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    :param R: 3x3 rotation matrix
    :return: quaternion as [x, y, z, w]
    """
    q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
    q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
    q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    
    return [q0, q1, q2, q3]

def orientation_cost_function(orientation):
    
    theta = np.linalg.norm(orientation)
    axis = orientation/theta if theta != 0 else [0,0,1]
    orientation = np.concatenate((axis, [theta if theta != 0 else 2*np.pi]))

    quaternion2 = normalize_quaternion(axis_angle_to_quaternion(orientation[0:3],orientation[3]))
    
    # change cad orientation
    depth_map2, object_pixels2 = generate_depth_map(object_name,translation_cad, quaternion2)
    obj_depth_image2 = crop_object_image(depth_map2,object_pixels2)
    obj_depth_image2_normalized = normalize_depth_map(obj_depth_image2)
    
    # resize images to the same size
    resized_image1_array, resized_image2_array = resize_images_to_same_size(obj_depth_image_normalized, obj_depth_image2_normalized)
        
    # compute cost
    aspect_ratio_1 = obj_depth_image_normalized.shape[1] / obj_depth_image_normalized.shape[0]
    aspect_ratio_2 = obj_depth_image2_normalized.shape[1] / obj_depth_image2_normalized.shape[0]

    h_image,w_image = resized_image1_array.shape   
    # cost = np.nansum(np.where(np.isnan(resized_image1_array) & np.isnan(resized_image2_array), 0, 
    #                           np.where(np.isnan(resized_image1_array) | np.isnan(resized_image2_array), 1, 
    #                                    1*pow((resized_image1_array - resized_image2_array),2))))
    
    cost = 0
    for w in range(w_image):
        for h in range(h_image):
            d1 = resized_image1_array[h][w]
            d2 = resized_image2_array[h][w]
                        
            if math.isnan(d1) and math.isnan(d2):  
                cost = cost + 0 # is good
            else:
                if math.isnan(d2) or math.isnan(d1):
                    cost = cost + 1
                else:
                    # print("depth_values", d1,d2)
                    cost = cost + pow((d1-d2),2)
    
    
    cost = cost/(h_image*w_image) + 0*abs(aspect_ratio_1-aspect_ratio_2)
    
    # print("orientation", orientation)
    # print("cost value", cost)
    # print("aspect ratios", aspect_ratio_1, aspect_ratio_2)
    


    return cost, depth_map2, resized_image1_array, resized_image2_array

def euler_to_quaternion(euler_angles):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.
    
    Parameters:
    roll : float
        The roll (rotation around x-axis) in radians.
    pitch : float
        The pitch (rotation around y-axis) in radians.
    yaw : float
        The yaw (rotation around z-axis) in radians.
    
    Returns:
    q : numpy array
        The quaternion as [qx, qy, qz, qw].
    """
    # Compute the quaternion elements
    
    roll, pitch, yaw = euler_angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

def initialize_nvisii(interactive, camera_intrinsics, object_name, obj_file_path):
        
    nvisii.initialize(headless=not interactive, verbose=True)
    nvisii.disable_updates()
    # nvisii.disable_denoiser()

    fx,fy,cx,cy,width_,height_ = camera_intrinsics
    camera = nvisii.entity.create(
        name="camera",
        transform=nvisii.transform.create("camera"),
        camera=nvisii.camera.create_from_intrinsics(
            name="camera",
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width_,
            height=height_
        )
    )
    camera.get_transform().look_at(
        at=(0, 0, 0),
        up=(0, -1, -1),
        eye=(1, 1, 0)
    )
    nvisii.set_camera_entity(camera)

    obj_mesh = nvisii.entity.create(
        name=object_name,
        mesh=nvisii.mesh.create_from_file(object_name, obj_file_path),
        transform=nvisii.transform.create(object_name),
        material=nvisii.material.create(object_name)
    )

    obj_mesh.get_transform().set_parent(camera.get_transform())

    nvisii.sample_pixel_area(
        x_sample_interval=(.5, .5),
        y_sample_interval=(.5, .5))
        
    
    return 

def change_object_mesh(old_object_name, new_object_name,obj_file_path):
    
    camera = nvisii.entity.get("camera")
   
    nvisii.entity.remove(old_object_name)
    
    obj_mesh = nvisii.entity.create(
        name=object_name,
        mesh=nvisii.mesh.create_from_file(new_object_name, obj_file_path),
        transform=nvisii.transform.create(new_object_name),
        material=nvisii.material.create(new_object_name)
    )

    obj_mesh.get_transform().set_parent(camera.get_transform())

    
    return 

def convert_from_uvd(h, w, d, fx, fy, cx, cy):
    px = (w - cx)/fx
    py = (h - cy)/fy
    
    z = d/np.sqrt(1. + px**2 + py**2)     
    return z

def generate_depth_map(object_name, position, quaternion):
    
    obj_mesh = nvisii.entity.get(object_name)

    # rotation camera color frame to nvisii frame Rx(pi)
    x = position[0]
    y = -position[1]
    z = -position[2]

    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]


    p = np.array([x, y, z])

    obj_mesh.get_transform().set_position(p)

    rotation =  nvisii.quat(qw,qx,qy,qz)
    rotation_flip = nvisii.angleAxis(-nvisii.pi(),nvisii.vec3(1,0,0)) * rotation # additional rotation due camera nvissi frame
    obj_mesh.get_transform().set_rotation(rotation_flip)
    
    obj_mesh.get_transform().set_scale(nvisii.vec3(mesh_scale))
        
    virtual_depth_array = nvisii.render_data(
        width=int(image_width),
        height=int(image_height),
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="depth"
    )

    virtual_depth_array = np.array(virtual_depth_array).reshape(image_height, image_width, 4)
    virtual_depth_array = np.flipud(virtual_depth_array)
    
    
    # # get pixels
    # object_pixels = []
    # for h in range(image_height):
    #         for w in range(image_width):
    #             if virtual_depth_array[h, w, 0] < max_virtual_depth and virtual_depth_array[h, w, 0] > 0:
    #                  object_pixels.append((h, w))
                
                     
    #             # fix also error in virtual depth nvisii
    #             if virtual_depth_array[h, w, 0] > max_virtual_depth or virtual_depth_array[h, w, 0] <= 0:
    #                 virtual_depth_array[h, w, 0] = np.nan
    #             else:
    #                 virtual_depth_array[h, w, 0] = convert_from_uvd(h, w, virtual_depth_array[h, w, 0], focal_length_x, focal_length_y, principal_point_x, principal_point_y)

    # Create a mask for the condition
    mask = (virtual_depth_array[:,:,0] < max_virtual_depth) & (virtual_depth_array[:,:,0] > 0)

    # Get the indices where the mask is True
    object_pixels = np.argwhere(mask)

    # Apply the conditions to the virtual_depth_array using the mask
    virtual_depth_array[mask, 0] = convert_from_uvd(object_pixels[:,0], object_pixels[:,1], virtual_depth_array[mask, 0], focal_length_x, focal_length_y, principal_point_x, principal_point_y)
    virtual_depth_array[~mask, 0] = np.nan
    object_pixels = list(map(tuple, object_pixels))

    return virtual_depth_array[:,:,0], object_pixels

def depth_to_pointcloud(depth_map, camera_intrinsics, object_pixels):
    # generate point cloud from depth_map
    # px = (X − cx)pz /fx, py = (Y − cy )pz /y
    
    height, width = depth_map.shape
    fx,fy,cx,cy,_,_ = camera_intrinsics
    
    pixel_h , pixel_w = map(list, zip(*object_pixels))
    max_w = max(pixel_w)
    max_h = max(pixel_h)
    min_w = min(pixel_w)
    min_h = min(pixel_h)
    new_pixel_w = np.linspace(min_w,max_w, width).astype(int)
    new_pixel_h = np.linspace(min_h,max_h, height).astype(int)


    point_cloud = []    
    for h in range(height):
        for w in range(width):
            if(math.isnan(depth_map[h][w])):
                continue
            pz = depth_map[h][w]
            w_ = new_pixel_w[w]
            h_ = new_pixel_h[h]
            px = ((w_ - cx)*pz)/fx
            py = ((h_ - cy)*pz)/fy
            point_cloud.append((px,py,pz))
                            
    return point_cloud

def depth_to_pointcloud_fromlist(depth_map, camera_intrinsics):
    # generate point cloud from depth_map
    # px = (X − cx)pz /fx, py = (Y − cy )pz /y
    fx,fy,cx,cy,width_,height_ = camera_intrinsics

    point_cloud = []    
    for h,w,dz in depth_map:
         pz = dz
         px = ((w - cx)*pz)/fx
         py = ((h - cy)*pz)/fy
         point_cloud.append((px,py,pz))
                
    return point_cloud

def plot_pointcloud(point_cloud, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [point[0] for point in point_cloud]
    ys = [point[1] for point in point_cloud]
    zs = [point[2] for point in point_cloud]
    ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    # plt.show()
    plt.savefig(title + '.png')
    return ax
    
def resample_depth_map(depth_map, object_pixels,width,height):
    # this method resamples the object depth map identified by object_pixels width width and height
    pixel_h , pixel_w = map(list, zip(*object_pixels))
    max_w = max(pixel_w)
    max_h = max(pixel_h)
    min_w = min(pixel_w)
    min_h = min(pixel_h)
        

    new_pixel_w = np.linspace(min_w,max_w, width).astype(int)
    new_pixel_h = np.linspace(min_h,max_h, height).astype(int)

    resampled_depth_map = []

    for h in new_pixel_h:
        for w in new_pixel_w:
            #if not math.isnan(depth_map[y,x]):
            resampled_depth_map.append((h,w,depth_map[h,w])) 
            
    #print(resampled_depth_map)
    return resampled_depth_map

def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t

def axis_angle_to_quaternion(axis, angle):
    axis = axis / np.linalg.norm(axis)  # Ensure the axis is a unit vector
    half_angle = angle / 2
    w = np.cos(half_angle)
    x, y, z = np.sin(half_angle) * axis
    return np.array([w, x, y, z])

def compute_object_center(object_pixels, depth, camera_intrinsics):
    pixel_y , pixel_x = map(list, zip(*object_pixels))
    max_x = max(pixel_x)
    max_y = max(pixel_y)
    min_x = min(pixel_x)
    min_y = min(pixel_y)
    
    
    center_x = int((max_x + min_x)/2)
    center_y = int((max_y + min_y)/2)
    
    fx,fy,cx,cy,width_,height_ = camera_intrinsics
   
    px = ((center_x - cx)*depth)/fx
    py = ((center_y - cy)*depth)/fy
    object_center = [px,py,depth]
    
    return object_center

def find_rotation_matrix(A,B):
    n, m = A.shape
    H = A.T @ B
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(H))
    S = np.diag([1] * (m - 1) + [d])

    
    R = U @ S @ VT    

    return R

def axis_angle_to_rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)  # Ensure the axis is a unit vector
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def rotation_matrix_to_axis_angle(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    axis = np.array([(R[2, 1] - R[1, 2]), (R[0, 2] - R[2, 0]), (R[1, 0] - R[0, 1])]) / (2 * np.sin(theta))
    return axis/np.linalg.norm(axis), theta

def normalize_rotation_matrix(R):
    U, S, VT = np.linalg.svd(R)
    return U @ VT

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

def icp(a, b, init_pose=(0,0,0), init_rotation = (1,0,0,0,1,0,0,0,1), no_iterations = 13):
    '''
    The Iterative Closest Point method: aligns two point clouds
    Parameters:
        a: Nxm numpy array of source mD points
        b: Nxm numpy array of destination mD point
        init_pose: (sx,sy,sz) initial pose
        no_iterations: number of iterations to run
    Returns:
        T: final homogeneous transformation that maps a on to b
        distances: Euclidean distances (errors) of the nearest neighbor
    '''


    src = np.array(a).T
    dst = np.array(b).T

    # Initialise overall_T with init_rotation and init_pose
    overall_T = np.eye(4)
    overall_T[0:3, 0:3] = np.array(init_rotation).reshape(3,3)
    overall_T[0:3, 3] = np.array(init_pose)

    #Bring the source cloud to the initial pose
    src = np.dot(overall_T, np.vstack((src, np.ones((1, src.shape[1])))))

    prev_error = 0

    for i in range(no_iterations):
        #Find the nearest neighbours in the destination cloud
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[:3,:].T)
        distances, indices = nbrs.kneighbors(src[:3,:].T)

        #Compute the transformation between the current source and nearest destination points
        indices_1D = indices.T.reshape(-1)

        R = find_rotation_matrix(dst[:3,indices_1D].T,src[:3,:].T)
        T = np.eye(4)
        T[0:3, 0:3] = (R)
        T[0:3, 3] = np.array([0,0,0])
        
        
        # Update the overall transformation
        overall_T = np.dot(T, overall_T)

        #Update the current source
        src = np.dot(T, src)
        
        # T, src_transformed, mean_error = trimesh.registration.procrustes(dst[:3,indices_1D].T,src[:3,:].T)
        # overall_T = np.dot(T, overall_T)
        # src = src_transformed

        mean_error = np.mean(distances)
        print("Mean error:", mean_error)
        if np.abs(prev_error - mean_error) < 0.000001:
            break
        prev_error = mean_error
    

    #Transform the original source cloud (not the working copy used in the iterations)
    print("Final transformation:", overall_T)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # xs = [point[0] for point in src.T]
    # ys = [point[1] for point in src.T]
    # zs = [point[2] for point in src.T]
    # ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    # xs = [point[0] for point in dst.T]
    # ys = [point[1] for point in dst.T]
    # zs = [point[2] for point in dst.T]
    # ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    # # plt.show()
    # plt.savefig("overlap_" + str(int(random.uniform(0,1000))) + '.png')
  
    rot_pcl = src[:3,:]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[:3,:].T)
    distances, indices = nbrs.kneighbors(src[:3,:].T)
    return overall_T, distances, rot_pcl, indices.T.reshape(-1), np.mean(distances)


def translate_centroid_to_origin(pcl):
    # Convert list to numpy array if it's not
    if isinstance(pcl, list):
        pcl = np.array(pcl)

    # Compute the centroid
    centroid = pcl.mean(axis=0)

    # Translate the point cloud
    translated_pcl = pcl - centroid

    return translated_pcl

def normalize_point_cloud(pcl):
    # Convert list to numpy array if it's not
    if isinstance(pcl, list):
        pcl = np.array(pcl)

    # # Compute the norm of each point
    # norms = np.linalg.norm(pcl, axis=1)

    # # Normalize the point cloud
    # normalized_pcl = pcl / norms[:, np.newaxis]
    
    
    # Compute the mean of the point cloud
    mean = np.mean(pcl, axis=0)

    # Compute the root mean square distance
    #rmsd = np.sqrt(np.mean(np.sum((pcl - mean)**2, axis=1)))
    rmsd = np.sqrt(np.mean(pcl**2))
    # Scale the point cloud by dividing each coordinate by the rmsd
    normalized_pcl = pcl / rmsd
    
    rmsd = np.sqrt((pcl ** 2).sum() / len(pcl))
    
    return normalized_pcl


############## MAIN CODE ############################



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
camera_intrinsic = [focal_length_x,focal_length_y,principal_point_x,principal_point_y,image_width,image_height]


# Load file real object
object_name = "banana"
file_name = "cad_models/rubber_duck_toy_1k.gltf"  
mesh_scale_real = 1 #0.01 banana
max_virtual_depth = 5 #[m]
mesh_scale = mesh_scale_real


# Pose real object
translation_real = np.array([0,-0,1]) # position of the object in meters wrt camera
euler_angles = [3,3.24,0] # radians - roll pitch and yaw
import random
euler_angles = [random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi)]
quaternion_real = euler_to_quaternion(euler_angles)#[0,0.5,0.5,0]  

# initialize nvisii
interactive = False
initialize_nvisii(interactive, camera_intrinsic,object_name, file_name)


# Generate the real depth map
depth_map, object_pixels = generate_depth_map(object_name,translation_real, quaternion_real) # The first time call it two times due to nvisii bug
depth_map, object_pixels = generate_depth_map(object_name,translation_real, quaternion_real)
# crop object image
obj_depth_image = crop_object_image(depth_map,object_pixels)

# normalize object depth map
obj_depth_image_normalized = normalize_depth_map(obj_depth_image)


# change object mesh to simulate differences between real object and cad
new_object_name = "banana2"
new_object_path = file_name #"cad_models/banana.obj"
mesh_scale_cad = mesh_scale_real*1
mesh_scale = mesh_scale_cad # change mesh scale to test different scales

change_object_mesh(object_name, new_object_name, new_object_path)
translation_cad = compute_object_center(object_pixels, 1.3, camera_intrinsic)


root = tk.Tk()

slider1 = tk.Scale(root, from_=-2*np.pi, to=2*np.pi, resolution=0.01, orient=tk.HORIZONTAL, label="Slider 1", length=400, command=evaluate_cost)
slider1.pack()

slider2 = tk.Scale(root, from_=-2*np.pi, to=2*np.pi, resolution=0.01, orient=tk.HORIZONTAL, label="Slider 2", length=400, command=evaluate_cost)
slider2.pack()

slider3 = tk.Scale(root, from_=-2*np.pi, to=2*np.pi, resolution=0.01, orient=tk.HORIZONTAL, label="Slider 3", length=400, command=evaluate_cost)
slider3.pack()

button = tk.Button(root, text="evaluate_cost", command=evaluate_cost)
button.pack()

root.mainloop()