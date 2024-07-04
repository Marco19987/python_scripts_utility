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
#         theta = np.linalg.norm(orientation)axis_angle_to_quaternion
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
    
    # theta = np.linalg.norm(orientation)
    # axis = orientation/theta if theta != 0 else [0,0,1]
    # orientation = np.concatenate((axis, [theta if theta != 0 else 2*np.pi]))
    # quaternion2 = normalize_quaternion(axis_angle_to_quaternion(orientation[0:3],orientation[3]))
    
    # euler angle
    quaternion2 = normalize_quaternion(euler_to_quaternion(orientation))
    
    # rot mstrix
    #orientation = orientation.reshape(3,3)
    #quaternion2 = rotation_matrix_to_quaternion(normalize_rotation_matrix(orientation))
    
    # continuos representation
    # orientation = orientation.reshape(3,2)
    # orientation = continuos_representation(orientation)
    # quaternion2 = rotation_matrix_to_quaternion(normalize_rotation_matrix(orientation))

    
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
    tmp1 = 0
    tmp2 = 0
    tmp3 = 0
    for w in range(w_image):
        for h in range(h_image):
            d1 = resized_image1_array[h][w]
            d2 = resized_image2_array[h][w]
                        
            if math.isnan(d1) and math.isnan(d2):  
                cost = cost + 0.5 # is good
                tmp1 = tmp1 + 0.5
            else:
                if math.isnan(d2) or math.isnan(d1):
                    cost = cost + 1
                    tmp2 = tmp2 + 1
                else:
                    # print("depth_values", d1,d2)
                    cost = cost + 0*pow((d1-d2),2) + 1*abs(d1-d2)
                    tmp3 = tmp3 + 0*pow((d1-d2),2) + 1*abs(d1-d2)
                    
    print("both nan", tmp1/(h_image*w_image))
    print("one nan", tmp2/(h_image*w_image))
    print("both not nan", tmp3/(h_image*w_image))
    
    cost = cost/(h_image*w_image) + 0*abs(aspect_ratio_1-aspect_ratio_2)
    
    # Initiate ORB detector
    # img1 = cv2.normalize(obj_depth_image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # img2 = cv2.normalize(obj_depth_image2, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # orb = cv2.ORB_create()

    # # Find the keypoints and descriptors with ORB
    # kp1, des1 = orb.detectAndCompute(img1,None)
    # kp2, des2 = orb.detectAndCompute(img2,None)

    # # Create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # # Match descriptors
    # matches = bf.match(des1,des2)

    # # Sort them in the order of their distance
    # matches = sorted(matches, key = lambda x:x.distance)

    # # Draw first 10 matches
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('Matches',img3)
    
    # # get point cloud
    # pcl_object_2 = np.array(depth_to_pointcloud(obj_depth_image2, camera_intrinsic, object_pixels))
    # pcl_object_2_normalized = normalize_point_cloud(translate_centroid_to_origin((pcl_object_2)))
    
    # # compute svd for pcl_object_1 and pcl_object_2
    # U1, S1, V1 = np.linalg.svd(pcl_object_real_normalized, full_matrices=False)
    # U2, S2, V2 = np.linalg.svd(pcl_object_2_normalized, full_matrices=False)
    
    # # compute cost ad the difference between the singular values
    # cost = np.linalg.norm(S1 - S2)
    
    # # compute angles between the singular vectors v1 and v2
    # theta_autovector_1 = np.arctan2(np.linalg.norm(np.cross(V1[0], V2[0])), np.dot(V1[0], V2[0]))
    # theta_autovector_2 = np.arctan2(np.linalg.norm(np.cross(V1[1], V2[1])), np.dot(V1[1], V2[1]))
    # theta_autovector_3 = np.arctan2(np.linalg.norm(np.cross(V1[2], V2[2])), np.dot(V1[2], V2[2]))
    # print("theta_autovector_1", theta_autovector_1)
    # print("theta_autovector_2", theta_autovector_2)
    # print("theta_autovector_3", theta_autovector_3)
    
    # cost = cost + (theta_autovector_1**2 + theta_autovector_2**2 + theta_autovector_3**2)
    
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # xs = [point[0] for point in pcl_object_2_normalized]
    # ys = [point[1] for point in pcl_object_2_normalized]
    # zs = [point[2] for point in pcl_object_2_normalized]
    # ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    # xs = [point[0] for point in pcl_object_real_normalized]
    # ys = [point[1] for point in pcl_object_real_normalized]
    # zs = [point[2] for point in pcl_object_real_normalized]
    # ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    # # orthogonal view plan xy
    # # ax.view_init(elev=90, azim=45)
    # # plot also the V1 and V2 frames
    # ax.quiver(0, 0, 0, V1[0][0], V1[0][1], V1[0][2], color='r')
    # ax.quiver(0, 0, 0, V1[1][0], V1[1][1], V1[1][2], color='g')
    # ax.quiver(0, 0, 0, V1[2][0], V1[2][1], V1[2][2], color='b')
    # ax.quiver(0, 0, 0, V2[0][0], V2[0][1], V2[0][2], color='r')
    # ax.quiver(0, 0, 0, V2[1][0], V2[1][1], V2[1][2], color='g')
    # ax.quiver(0, 0, 0, V2[2][0], V2[2][1], V2[2][2], color='b')
    
    # plt.savefig("cost_pcl" + '.png')
    
    
    # print("Singular values", S1, S2)
    
    print("orientation", orientation)
    print("cost value", cost)
    print("aspect ratios", aspect_ratio_1, aspect_ratio_2)
    
    cv2.imshow("depth_map2", depth_map2)
    cv2.imshow("Real image", resized_image1_array)
    cv2.imshow("Cad model", resized_image2_array)
    cv2.waitKey(0)

    return cost

def test_rotation_optimization(guess):
    
    U1, S1, V1 = np.linalg.svd(pcl_object_real_normalized, full_matrices=False)

    cost = 1000
    orientation = guess
    U2, S2, V2 = [],[],[]
    Rold = quaternion_to_rotation_matrix(orientation)
    Vold = np.eye(3)
    while cost>0.1:
        # euler angle
        quaternion2 = normalize_quaternion((orientation))
        
        # change cad orientation
        # depth_map2, object_pixels2 = generate_depth_map(object_name,translation_cad, quaternion2)
        # obj_depth_image2 = crop_object_image(depth_map2,object_pixels2)
        
        # # get point cloud
        # pcl_object_2 = np.array(depth_to_pointcloud(obj_depth_image2, camera_intrinsic, object_pixels))
        # pcl_object_2_normalized = normalize_point_cloud(translate_centroid_to_origin((pcl_object_2)))
        
        Rcad = quaternion_to_rotation_matrix(quaternion2)
        pcl_obj_tmp = Rcad @ pcl_obj.T 
        pcl_object_2_normalized = (translate_centroid_to_origin(pcl_obj_tmp.T))
        
        # compute svd for pcl_object_1 and pcl_object_2
        U2, S2, V2 = np.linalg.svd(pcl_object_2_normalized, full_matrices=False)
        
        # compute cost ad the difference between the singular values
        cost = np.linalg.norm(S1 - S2)
        
        # compute angles between the singular vectors v1 and v2
        theta_autovector_1 = np.arctan2(np.linalg.norm(np.cross(V1[0], V2[0])), np.dot(V1[0], V2[0]))
        theta_autovector_2 = np.arctan2(np.linalg.norm(np.cross(V1[1], V2[1])), np.dot(V1[1], V2[1]))
        theta_autovector_3 = np.arctan2(np.linalg.norm(np.cross(V1[2], V2[2])), np.dot(V1[2], V2[2]))
        print("theta_autovector_1", theta_autovector_1)
        print("theta_autovector_2", theta_autovector_2)
        print("theta_autovector_3", theta_autovector_3)
        
        cost = cost + (theta_autovector_1**2 + theta_autovector_2**2 + theta_autovector_3**2)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = [point[0] for point in pcl_object_2_normalized]
        ys = [point[1] for point in pcl_object_2_normalized]
        zs = [point[2] for point in pcl_object_2_normalized]
        ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        # xs = [point[0] for point in pcl_object_real_normalized]
        # ys = [point[1] for point in pcl_object_real_normalized]
        # zs = [point[2] for point in pcl_object_real_normalized]
        ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        ax.view_init(elev=0, azim=45)
        # ax.quiver(0, 0, 0, V1[0][0], V1[1][0], V1[2][0], color='r')
        # ax.quiver(0, 0, 0, V1[0][1], V1[1][1], V1[2][1], color='g')
        # ax.quiver(0, 0, 0, V1[0][2], V1[1][2], V1[2][2], color='b')
        ax.quiver(0, 0, 0, V2[0][0], V2[1][0], V2[2][0], color='r', label='V2 cad model')
        ax.quiver(0, 0, 0, V2[0][1], V2[1][1], V2[2][1], color='g', label='V2 cad model')
        ax.quiver(0, 0, 0, V2[0][2], V2[1][2], V2[2][2], color='b', label='V2 cad model')
        # ax.quiver(0, 0, 0, Vold[0][0], Vold[1][0], Vold[2][0], color='r', label='V2 rotated')
        # ax.quiver(0, 0, 0, Vold[0][1], Vold[1][1], Vold[2][1], color='g', label='V2 rotated')
        # ax.quiver(0, 0, 0, Vold[0][2], Vold[1][2], Vold[2][2], color='b', label='V2 rotated')
        plt.savefig("cost_pcl" + '.png')
        

        # Compute the rotation matrix as rotx(theta_autovector_1) roty(theta_autovector_2) rotz(theta_autovector_3)
        Rotx = np.array([[1, 0, 0], [0, np.cos(theta_autovector_1), -np.sin(theta_autovector_1)], [0, np.sin(theta_autovector_1), np.cos(theta_autovector_1)]])
        Roty = np.array([[np.cos(theta_autovector_2), 0, np.sin(theta_autovector_2)], [0, 1, 0], [-np.sin(theta_autovector_2), 0, np.cos(theta_autovector_2)]])
        Rotz = np.array([[np.cos(theta_autovector_3), -np.sin(theta_autovector_3), 0], [np.sin(theta_autovector_3), np.cos(theta_autovector_3), 0], [0, 0, 1]])
        #Rot = np.dot(Rotz, np.dot(Roty, Rotx))
        
        # find a versor perpendicular to the two vectors v1(:,1) and v2(:,1)
        
        v1 = V1[:,0]
        v2 = V2[:,0]
        v = np.cross(v1, v2)
        v = v/np.linalg.norm(v)
        # compute the angle between the two vectors
        theta_autovector_1 = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
        theta = theta_autovector_1
        
     

        # compute the rotation matrix
        #Rot = np.eye(3) + np.sin(theta)*np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) + (1 - np.cos(theta))*np.dot(v[:, None], v[None, :])
        Rot = axis_angle_to_rotation_matrix(v, theta)
        
        

        
        # do the same for the second vector
        v1 = V1[:,1]
        v2 = V2[:,1]
        v = np.cross(v1, v2)
        v = v/np.linalg.norm(v)
        # # compute the angle between the two vectors
        theta_autovector_2 = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
    

        theta = theta_autovector_2
        # # compute the rotation matrix
        # Rot = axis_angle_to_rotation_matrix(v, theta) @ Rot 
       
        Rot2 = axis_angle_to_rotation_matrix([1.0,0,0],theta) 
          

        # # do the same for the third vector
        v1 = V1[2]
        v2 = V2[2]
        v = np.cross(v1, v2)
        v = v/np.linalg.norm(v)
        # # compute the angle between the two vectors
        theta_autovector_3 = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
        theta = theta_autovector_3
        # # compute the rotation matrix
        #Rot = axis_angle_to_rotation_matrix(v, theta) @ Rot

       
        
        Rot =  Rot @ Rold  
        Rold = Rot  
        
        #Vold = Rot @ V2 @ Rot2
        print("R", normalize_rotation_matrix(Rot))
        orientation = rotation_matrix_to_quaternion(normalize_rotation_matrix(Rot))
        

        print("Singular values", S1, S2)
        
        print("orientation", orientation)
        print("cost value", cost)
        
        # cv2.imshow("Real image", obj_depth_image_normalized)
        # cv2.imshow("Cad model", normalize_depth_map(obj_depth_image2))
        cv2.waitKey(0)

    return cost







def euler_to_quaternion(euler_angles, sequence='ZYZ'):
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
    if sequence == 'XYZ':
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
    if sequence == 'ZYZ':
        alpha, beta, gamma = euler_angles
        ca = np.cos(alpha / 2)
        sa = np.sin(alpha / 2)
        cb = np.cos(beta / 2)
        sb = np.sin(beta / 2)
        cg = np.cos(gamma / 2)
        sg = np.sin(gamma / 2)

        qw = ca * cb * cg - sa * sb * sg
        qx = ca * cb * sg + sa * sb * cg
        qy = ca * sb * cg - sa * cb * sg
        qz = ca * sb * sg + sa * cb * cg

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
    U_, S_, VT_ = np.linalg.svd(R)
    return U_ @ VT_

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
        # print("Mean error:", mean_error)
        if np.abs(prev_error - mean_error) < 0.000001:
            break
        prev_error = mean_error
    

    #Transform the original source cloud (not the working copy used in the iterations)
    # print("Final transformation:", overall_T)
    
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

    # divide pcl by the module of the point with the maximum module
    normalized_pcl = pcl / np.linalg.norm(pcl, axis=1).max()
    
    return normalized_pcl


def continuos_representation(A):
    b1 = A[:, 0]
    b1 = b1 / np.linalg.norm(b1)
    b2 = A[:, 1] - np.dot(b1.T, A[:, 1]) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    B = np.column_stack((b1, b2, b3))
    return B


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
file_name = "cad_models/rubber_duck_toy.obj"  
mesh_scale_real = 1 #0.01 banana
max_virtual_depth = 5 #[m]
mesh_scale = mesh_scale_real


# Pose real object
translation_real = np.array([0.1,-0.1,0.9]) # position of the object in meters wrt camera
euler_angles = [0,0,0] # radians - roll pitch and yaw
# import random
# euler_angles = [random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi)]
quaternion_real = euler_to_quaternion(euler_angles)#[0,0.5,0.5,0]  

# initialize nvisii
interactive = False
initialize_nvisii(interactive, camera_intrinsic,object_name, file_name)


# Generate the real depth map
depth_map, object_pixels = generate_depth_map(object_name,translation_real, quaternion_real) # The first time call it two times due to nvisii bug
depth_map, object_pixels = generate_depth_map(object_name,translation_real, quaternion_real)
#cv2.imshow("depth_map", depth_map)
# crop object image
obj_depth_image = crop_object_image(depth_map,object_pixels)

# normalize object depth map
obj_depth_image_normalized = normalize_depth_map(obj_depth_image)


# change object mesh to simulate differences between real object and cad
new_object_name = "banana2"
new_object_path = "cad_models/rubber_duck_toy.obj"
mesh_scale_cad = mesh_scale_real*0.5
mesh_scale = mesh_scale_cad # change mesh scale to test different scales

change_object_mesh(object_name, new_object_name, new_object_path)
translation_cad = compute_object_center(object_pixels, 0.5, camera_intrinsic)

# cad model point cloud
pcl_obj, faces = read_obj_file(new_object_path)
pcl_obj_norm = normalize_point_cloud(translate_centroid_to_origin(pcl_obj*mesh_scale_cad))
plot_pointcloud(pcl_obj_norm, "pcl_obj_norm")

# ICP - to find a good initial guess
pcl_object_real = np.array(depth_to_pointcloud(obj_depth_image, camera_intrinsic, object_pixels))
pcl_object_real_normalized = normalize_point_cloud(translate_centroid_to_origin((pcl_object_real)))

plot_pointcloud(pcl_object_real_normalized, "pcl_object_normalized")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = [point[0] for point in pcl_obj_norm]
ys = [point[1] for point in pcl_obj_norm]
zs = [point[2] for point in pcl_obj_norm]
ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
xs = [point[0] for point in pcl_object_real_normalized]
ys = [point[1] for point in pcl_object_real_normalized]
zs = [point[2] for point in pcl_object_real_normalized]
ax.scatter(xs, ys, zs), ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
ax.view_init(elev=20, azim=30)
plt.savefig("overlap" + '.png')

R_init  = axis_angle_to_rotation_matrix([0,0,1],0)
transformation, distances, rotated_pcl, indices, mean_error = icp(np.array(pcl_obj_norm),np.array(pcl_object_real_normalized), init_pose=(0,0,0),init_rotation=R_init, no_iterations = 30)


axis, theta = rotation_matrix_to_axis_angle(normalize_rotation_matrix(transformation[:3,:3]))
initial_guess_icp = axis*theta

# Optimization of the orientation 

# Define your list of initial guesses
initial_guesses = [[0,0,0],
                   [np.pi/2,np.pi/2,np.pi/2], [np.pi, np.pi, np.pi] ,[3*np.pi/2,3*np.pi/2,3*np.pi/2],
                   [np.pi/2,np.pi/2,0], [0,np.pi/2,np.pi/2],[np.pi/2,0,np.pi/2],
                   [np.pi/2, 0, 0], [np.pi, 0, 0], [3*np.pi/2, 0, 0], 
                   [0, np.pi/2, 0], [0, np.pi, 0], [0, 3*np.pi/2, 0], 
                   [0, 0, np.pi/2], [0, 0, np.pi], [0, 0, 3*np.pi/2]]
initial_guesses = [[0,1.57, 0]]
# Rstart = transformation[:3,:3]

# initial_guesses_tmp = []
# for guess in initial_guesses:
#     if np.linalg.norm(guess) != 0:
#         axis = guess/np.linalg.norm(guess) 
#     else:
#         axis = [0,0,1]
#     theta = np.linalg.norm(guess)
#     Ratt = axis_angle_to_rotation_matrix(axis,theta) # additional rotation from the Rstart
#     axis_, theta_ = rotation_matrix_to_axis_angle(normalize_rotation_matrix(Rstart @ Ratt))
#     initial_guesses_tmp.append(axis_*theta_)
# initial_guesses = initial_guesses_tmp



#initial_guesses = [initial_guess_icp]


#initial_guesses = [[1,0,0,1,0,0]]
#initial_guesses = [0,0,0]
bnds = [(-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi, np.pi)]
module_constraint = NonlinearConstraint(lambda x: np.linalg.norm(x), 0, 2*np.pi)

def optimize(guess):
    return minimize(orientation_cost_function, guess,method="Powell", bounds=bnds, options={'ftol': 1e-3,'eps':1e-5})
    return minimize(orientation_cost_function, guess,method="Powell", tol = 1e-3, bounds=bnds, options={'ftol': 1e-3,'xtol' : 1e-1, 'eps': 1.57,'maxiter': 30,'disp': True})#, constraints=module_constraint)
    theta = np.linalg.norm(guess)
    axis = guess[0:3]/theta if theta != 0 else [0,0,1]
    R_init = axis_angle_to_rotation_matrix(axis,theta)
    transformation, distances, rotated_pcl, indices, mean_error = icp(np.array(pcl_obj_norm),np.array(pcl_object_real_normalized), init_pose=(0,0,0),init_rotation=R_init, no_iterations = 30)
    axis, theta = rotation_matrix_to_axis_angle(normalize_rotation_matrix(transformation[:3,:3]))
    # create a struct result with fields success, x, fun, message
    result = OptimizeResult()
    result.success = True
    result.x = axis*theta
    result.fun = mean_error
    result.message = "Optimization terminated successfully."
    return result
 
 
#test_rotation_optimization([0,0,0,1])    
result = optimize(initial_guesses)
# optimize(result.x)

# try evloutional algorithm
#result = differential_evolution(orientation_cost_function, bounds=bnds, maxiter=10, popsize=10, disp=True, workers=1, updating='deferred')

#Create a ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=20) as executor:
    # Use the executor to map the optimize function to the initial guesses
    results = executor.map(optimize, initial_guesses)

# results now contains the result of the optimization for each initial guess
result_cost = []
results_array = []
for result in results:
    print("Success:" + str(result.success))
    print("Result:" + str(result.x))
    print("Cost:" + str(result.fun))
    print("Message:" + str(result.message))
    result_cost.append(result.fun)
    results_array.append(result.x)


# Elaborate result after optimization

# # get pose cad model
# cont = 0
# for result in results_array:
#     orientation2 = result
#     theta = np.linalg.norm(orientation2)
#     axis = orientation2/theta
#     orientation2 = np.concatenate((axis, [theta]))
#     quaternion2 = axis_angle_to_quaternion(orientation2[0:3],orientation2[3])
#     quaternion2 = normalize_quaternion(quaternion2)

#     # place it in the virtual world
#     depth_map2, object_pixels2 = generate_depth_map(object_name,translation_cad, quaternion2)
#     obj_depth_image2 = crop_object_image(depth_map2,object_pixels2)
#     obj_depth_image2_normalized = normalize_depth_map(obj_depth_image2)    
#     cv2.imshow("depth_map2", depth_map2)


#     # Retrieve objects point clouds and resample them to get two ordered point clouds
#     resized_image_real_object, resized_image_cad_model = resize_images_to_same_size(obj_depth_image, obj_depth_image2)
#     cv2.imshow("resized_image_real_object", normalize_depth_map(resized_image_real_object))
#     cv2.imshow("resized_image_cad_model", normalize_depth_map(resized_image_cad_model))
#     cv2.waitKey(0)    
#     print("result",result_cost[cont])
#     print("result",result)
#     cont = cont + 1
    
orientation2 = results_array[np.argmin(result_cost)]
# theta = np.linalg.norm(orientation2)
# axis = orientation2/theta
# orientation2 = np.concatenate((axis, [theta]))
# quaternion2 = axis_angle_to_quaternion(orientation2[0:3],orientation2[3])

# euler
quaternion2 = euler_to_quaternion(orientation2)

# continuos
# orientation2 = orientation2.reshape(3,2)
# orientation2 = continuos_representation(orientation2)
# quaternion2 = rotation_matrix_to_quaternion(normalize_rotation_matrix(orientation2))


quaternion2 = normalize_quaternion(quaternion2)

# place it in the virtual world
depth_map2, object_pixels2 = generate_depth_map(object_name,translation_cad, quaternion2)
obj_depth_image2 = crop_object_image(depth_map2,object_pixels2)
obj_depth_image2_normalized = normalize_depth_map(obj_depth_image2)    
cv2.imshow("depth_map2", depth_map2)


# Retrieve objects point clouds and resample them to get two ordered point clouds
resized_image_real_object, resized_image_cad_model = resize_images_to_same_size(obj_depth_image, obj_depth_image2)
cv2.imshow("resized_image_real_object", normalize_depth_map(resized_image_real_object))
cv2.imshow("resized_image_cad_model", normalize_depth_map(resized_image_cad_model))

res_height,res_width = resized_image_real_object.shape
resampled_depth_map_real = resample_depth_map(depth_map, object_pixels,res_width,res_height)
resampled_depth_map_cad = resample_depth_map(depth_map2, object_pixels2,res_width,res_height)

point_cloud_real = depth_to_pointcloud_fromlist(resampled_depth_map_real,camera_intrinsic)
point_cloud_cad = depth_to_pointcloud_fromlist(resampled_depth_map_cad,camera_intrinsic)

point_cloud_real = np.array(point_cloud_real)
point_cloud_cad = np.array(point_cloud_cad)

mask = ~np.isnan(point_cloud_real) & ~np.isnan(point_cloud_cad)
mask_matrix = mask.reshape(res_width*res_height,3)
nan_depth_index = mask_matrix[:,2]

point_cloud_real = point_cloud_real[nan_depth_index]
point_cloud_cad = point_cloud_cad[nan_depth_index]

plot_pointcloud(point_cloud_real,"point_cloud_real")
plot_pointcloud(point_cloud_cad,"point_cloud_cad")

# now we have two ordered point clouds, we can run Umeyama and retrieve the relative translation, orientation and scale 
R, c, t = kabsch_umeyama(point_cloud_real, point_cloud_cad)

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

R2 = quaternion_to_rotation_matrix(quaternion2)

estimated_p_real = t + c * R @ np.array(translation_cad).T
estimated_R_real = normalize_rotation_matrix(c * R @ R2)  #???
estimated_scale_real = mesh_scale_cad*c

print("real object position", estimated_p_real)
print("real object orientation", estimated_R_real) 
print("real object scale", estimated_scale_real)

cv2.waitKey(0)
cv2.destroyAllWindows()
nvisii.deinitialize()

