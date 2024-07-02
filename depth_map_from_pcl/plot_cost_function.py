import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


import pickle
import cv2
import numpy as np
import nvisii 
import math
from scipy.optimize import minimize, NonlinearConstraint
from concurrent.futures import ThreadPoolExecutor



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
    
    print("orientation", orientation)
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
    

    print("orientation", orientation)
    print("cost value", cost)
    print("aspect ratios", aspect_ratio_1, aspect_ratio_2)
    
    cv2.imshow("depth_map2", depth_map2)
    cv2.imshow("Real image", resized_image1_array)
    cv2.imshow("Cad model", resized_image2_array)
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
    
    
    # Create a mask for the condition
    mask = (virtual_depth_array[:,:,0] < max_virtual_depth) & (virtual_depth_array[:,:,0] > 0)

    # Get the indices where the mask is True
    object_pixels = np.argwhere(mask)

    # Apply the conditions to the virtual_depth_array using the mask
    virtual_depth_array[mask, 0] = convert_from_uvd(object_pixels[:,0], object_pixels[:,1], virtual_depth_array[mask, 0], focal_length_x, focal_length_y, principal_point_x, principal_point_y)
    virtual_depth_array[~mask, 0] = np.nan
    object_pixels = list(map(tuple, object_pixels))

    return virtual_depth_array[:,:,0], object_pixels



    
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


def compute_cost(resized_image1_array, resized_image2_array):


    h_image,w_image = resized_image1_array.shape   
    
    cost = 0
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
                   
    cost = cost/(h_image*w_image)
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
    
def resample_depth_map(depth_map_obj, object_pixels,width,height):
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
            resampled_depth_map.append((h,w,depth_map_obj[h,w])) 
            
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




# Load the pre-generated viewpoints from the file
viewpoint_file = 'rubber_duck_viewpoints_45.pkl'
visualize_viwepoints = False

with open(viewpoint_file, 'rb') as f:
    data = pickle.load(f)

# Iterate over each element in the data
if visualize_viwepoints:
    for i, element in enumerate(data):
        euler_angles = element['euler_angles']
        depth_map = element['depth_map']
        aspect_ratio = element['aspect_ratio']

        # Print the data for this element
        print(f'Element {i}:')
        print(f'Euler angles: {euler_angles}')
        print(f'Aspect ratio: {aspect_ratio}')
        print(f'Depth map shape: {depth_map.shape}')
        cv2.imshow('Depth map', depth_map)
        cv2.waitKey(100)
        print()
    
    

# Load real object file
object_name = "banana"
file_name = "cad_models/rubber_duck_toy_1k.gltf"  
mesh_scale_real = 1 #0.01 banana
max_virtual_depth = 5 #[m]
mesh_scale = mesh_scale_real


############## generate depth image of the real object ####################################################

# Pose real object
translation_real = np.array([0,0,1]) # position of the object in meters wrt camera
euler_angles = [0.5,0.8,0.3] # radians - roll pitch and yaw

# import random
# euler_angles = [random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi)]
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

############################################################################################################


###################### FIND INITIAL GUESS FOR THE ORIENTATION ################################################

#Use the viewpoints to find a good initial guess for the orientation

# The real aspect ratio
aspect_ratio_real = obj_depth_image.shape[1] / obj_depth_image.shape[0]

# Sort the data based on the absolute difference with the real aspect ratio
data.sort(key=lambda x: abs(x['aspect_ratio'] - aspect_ratio_real))

# Number of elements to select
N = 5000

# Select the first N elements
selected_data = data[:N]

# Iterate over each selected element
index_min = 0
cost_min = 1000
cost_list = []
for i, element in enumerate(selected_data):
    euler_angles = element['euler_angles']
    depth_map_cad = element['depth_map']
    aspect_ratio = element['aspect_ratio']
    
    res1, res2 = resize_images_to_same_size(obj_depth_image_normalized, depth_map_cad)
    cost = compute_cost(res1, res2)
    cost_list.append(cost)
    if cost <= cost_min:
        index_min = i
        cost_min = cost
        
    # if cost_min < 0.05:
    #     break
    
    
    # Print the data for this element
    print(f'Element {i}:')
    print(f'Euler angles: {euler_angles}')
    print(f'Aspect ratio: {aspect_ratio}')
    print("real aspect ratio", aspect_ratio_real)
    print(f'Depth map shape: {depth_map.shape}')
    print("cost", cost)    
    print()
    # cv2.imshow('Depth map', depth_map)
    # cv2.waitKey(0)

print("index_min", index_min)
print("cost_min", cost_min) 
depth_cad = selected_data[index_min]['depth_map']


cv2.imshow("real object", obj_depth_image_normalized)
cv2.imshow("cad object", depth_cad)
cv2.waitKey(0)
cv2.destroyAllWindows()

orientation2 = selected_data[index_min]['euler_angles']
print("initial guess", orientation2)


#################### plot cost function ########################################################################
# plot the cost function by varying the orientation
plt.plot(cost_list)
plt.xlabel('Orientation')
plt.ylabel('Cost')
plt.title('Cost function')
plt.savefig('cost_function.png')
plt.show()




from scipy.interpolate import griddata

# Create a figure with 3 subplots (3D plots)
fig = plt.figure(figsize=(18, 6))

# Prepare data for interpolation
euler_angles = np.array([element['euler_angles'] for element in selected_data])
costs = np.array([compute_cost(resize_images_to_same_size(obj_depth_image_normalized, element['depth_map'])[0],
                               resize_images_to_same_size(obj_depth_image_normalized, element['depth_map'])[1])
                  for element in selected_data])

# Define meshgrid resolution
grid_resolution = 100j

# Subplot 1: Fix the first orientation, vary the other two
ax1 = fig.add_subplot(131, projection='3d')
grid_x1, grid_y1 = np.mgrid[min(euler_angles[:,1]):max(euler_angles[:,1]):grid_resolution, min(euler_angles[:,2]):max(euler_angles[:,2]):grid_resolution]
grid_z1 = griddata(euler_angles[:,1:3], costs, (grid_x1, grid_y1), method='cubic')
ax1.plot_surface(grid_x1, grid_y1, grid_z1, cmap='viridis', edgecolor='none')
ax1.set_title('Fixed Orientation 1')
ax1.set_xlabel('Orientation 2')
ax1.set_ylabel('Orientation 3')
ax1.set_zlabel('Cost')

# Subplot 2: Fix the second orientation, vary the others
ax2 = fig.add_subplot(132, projection='3d')
grid_x2, grid_y2 = np.mgrid[min(euler_angles[:,0]):max(euler_angles[:,0]):grid_resolution, min(euler_angles[:,2]):max(euler_angles[:,2]):grid_resolution]
grid_z2 = griddata(euler_angles[:,[0,2]], costs, (grid_x2, grid_y2), method='cubic')
ax2.plot_surface(grid_x2, grid_y2, grid_z2, cmap='viridis', edgecolor='none')
ax2.set_title('Fixed Orientation 2')
ax2.set_xlabel('Orientation 1')
ax2.set_ylabel('Orientation 3')
ax2.set_zlabel('Cost')

# Subplot 3: Fix the third orientation, vary the others
ax3 = fig.add_subplot(133, projection='3d')
grid_x3, grid_y3 = np.mgrid[min(euler_angles[:,0]):max(euler_angles[:,0]):grid_resolution, min(euler_angles[:,1]):max(euler_angles[:,1]):grid_resolution]
grid_z3 = griddata(euler_angles[:,0:2], costs, (grid_x3, grid_y3), method='cubic')
ax3.plot_surface(grid_x3, grid_y3, grid_z3, cmap='viridis', edgecolor='none')
ax3.set_title('Fixed Orientation 3')
ax3.set_xlabel('Orientation 1')
ax3.set_ylabel('Orientation 2')
ax3.set_zlabel('Cost')

# in each subplot, plot the initial guess
ax1.scatter(orientation2[1], orientation2[2], cost_min, color='red', s=100)
ax2.scatter(orientation2[0], orientation2[2], cost_min, color='red', s=100)
ax3.scatter(orientation2[0], orientation2[1], cost_min, color='red', s=100)

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.savefig('cost_function_3d_continuous_all.png')
plt.show()



cv2.destroyAllWindows()
nvisii.deinitialize()