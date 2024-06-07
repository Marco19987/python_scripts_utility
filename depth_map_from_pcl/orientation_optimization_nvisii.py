import numpy as np
from scipy.optimize import minimize, dual_annealing,NonlinearConstraint, least_squares,basinhopping,differential_evolution
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import nvisii
import time 

def crop_object_image(depth_map,object_pixels):
    
    pixel_y , pixel_x = map(list, zip(*object_pixels))
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
    print(dimensions)
    
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
            
        
        resized_image_1 = cv2.resize(image1_array, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
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
            
                    
        resized_image_1 = cv2.resize(image2_array, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
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
            for y in range(disp, disp+res_dim[min_width_index]):
                 for x in range(res_height_1):
                     resized_image[x][y] = tmp_image[min_width_index][x][y-disp]
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
            for x in range(disp, disp+res_dim[min_height_index]):
                for y in range(res_width_1):                    
                    resized_image[x][y] = tmp_image[min_height_index][x-disp][y]
            resized_image_1 = tmp_image[max_height_index]
            resized_image_2 = resized_image
            if min_height_index == image_resized_index:
                switch_images = True
                
    if switch_images:
        tmp = resized_image_2
        resized_image_2 = resized_image_1
        resized_image_1 = tmp
            

    return resized_image_1, resized_image_2

def orientation_cost_function(orientation):
    #translation2 = [-0.2,-0.1,0.5]
        
    # if np.linalg.norm(orientation) != 0:
    #     theta = np.linalg.norm(orientation)
    #     axis = orientation/theta
    # else:
    #     theta = 0
    #     axis = [0,0,1]
        
    # print("axis", axis)
    # print("theta", theta)        
    # orientation = np.concatenate((axis, [theta]))
    

    
    quaternion2 = euler_to_quaternion(orientation)
    # quaternion2 = orientation
    #quaternion2 = axis_angle_to_quaternion(orientation[0:3],orientation[3])
    
    quaternion2 = normalize_quaternion(quaternion2)

    depth_map2, object_pixels2 = generate_depth_map(object_name,translation2, quaternion2)
   
    # # crop object image
    obj_depth_image2 = crop_object_image(depth_map2,object_pixels2)
    
    # # normalize object depth map
    obj_depth_image2_normalized = normalize_depth_map(obj_depth_image2)
    
    resized_image1_array, resized_image2_array = resize_images_to_same_size(obj_depth_image_normalized, obj_depth_image2_normalized)
        
    w1,h1 = obj_depth_image_normalized.shape
    w2,h2 = obj_depth_image2_normalized.shape
    
    aspect_ratio_1 = w1/h1
    aspect_ratio_2 = w2/h2
    
    
    #   # Compute the cost as the sum of squared differences of non-NaN pixels
    # mask = ~np.isnan(resized_image1_array) 
    # mask2 = ~np.isnan(resized_image1_array) 
    # #resized_image2_array[mask] = 5
    # #mask = ~np.isnan(resized_image1_array) & ~np.isnan(resized_image2_array)
    # #cost = np.nansum((resized_image1_array[mask] - resized_image2_array[mask]) ** 2)
    # cost = np.nansum((resized_image1_array - resized_image2_array) ** 2)
    # # cost = cost/(w1*h1)
    # cpost = cost/(mask.size + mask2.size)

    # #cost = cost/(resized_image1_array[mask].size)
    # #cost = cost + (aspect_ratio_1-aspect_ratio_2)**2
    

    
    # Compute the cost as the sum of squared differences of non-NaN pixels
    #mask = ~np.isnan(resized_image1_array) 
    #mask2 = ~np.isnan(resized_image2_array) 
    
    # occupied_pixels = np.copy(resized_image1_array)
    # occupied_pixels[mask] = 1
    # occupied_pixels[~mask] = 0
    
    
    # occupied_pixels_2 = np.copy(resized_image2_array)
    # occupied_pixels_2[mask2] = 1
    # occupied_pixels_2[~mask2] = 0
    
    # cv2.imshow("occupied_pixels", occupied_pixels)
    # cv2.imshow("occupied_pixels2", occupied_pixels_2)

    
    # resized_image1_array[~mask] = 10
    # resized_image2_array[~mask2] = 10
    # #mask = ~np.isnan(resized_image1_array) & ~np.isnan(resized_image2_array)
    # #cost = np.nansum((resized_image1_array[mask] - resized_image2_array[mask]) ** 2)
    # cost = np.nansum((resized_image1_array - resized_image2_array) ** 2)
    # #cost =  cost + np.sum((occupied_pixels-occupied_pixels_2)**2)
    # cost = cost/(w1*h1)
    # #cost = cost/(mask.size + mask2.size)

    # #cost = cost/(resized_image1_array[mask].size)
    # cost = 10*cost + 1*(aspect_ratio_1-aspect_ratio_2)**2

    
    w,h = resized_image1_array.shape
    cost = 0
    for i in range(w):
        for j in range(h):
            d1 = resized_image1_array[i][j]
            d2 = resized_image2_array[i][j]
                        
            if math.isnan(d1) and math.isnan(d2):  
                cost = cost + 0 # is good
            else:
                if math.isnan(d2) or math.isnan(d1):
                    cost = cost + 1
                else:
                    # print("depth_values", d1,d2)
                    cost = cost + pow((d1-d2),2)
                    

    # cost = cost/(w*h)
    cost = cost/(w*h)
    cost = cost + 0*(aspect_ratio_1-aspect_ratio_2)**2 
        
    
    print("orientation", orientation)
    print("cost value", cost)
    print("aspect ratios", aspect_ratio_1, aspect_ratio_2)
    
    # cv2.imshow("depth_map2", depth_map2)
    # cv2.imshow("Real image", resized_image1_array)
    # cv2.imshow("Cad model", resized_image2_array)
    # cv2.waitKey(0)

    return cost

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
    
    
    # get pixels
    object_pixels = []
    for i in range(image_height):
            for j in range(image_width):
                if virtual_depth_array[i, j, 0] < max_virtual_depth and virtual_depth_array[i, j, 0] > 0:
                     object_pixels.append((i, j))
                
                     
                # fix also error in virtual depth nvisii
                if virtual_depth_array[i, j, 0] > max_virtual_depth or virtual_depth_array[i, j, 0] <0:
                    virtual_depth_array[i, j, 0] = np.nan
                else:
                    virtual_depth_array[i, j, 0] = convert_from_uvd(i, j, virtual_depth_array[i, j, 0], focal_length_x, focal_length_y, principal_point_x, principal_point_y)

    return virtual_depth_array[:,:,0], object_pixels

    
def convert_from_uvd(v, u, d, fx, fy, cx, cy):
    x_over_z = (cx - u) / fx
    y_over_z = (cy - v) / fy
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    return z

def unit_quaternion_constraint(quaternion):
    return np.sum(np.square(quaternion)) - 1
    
def depth_to_pointcloud(depth_map, camera_intrinsics):
    # generate point cloud from depth_map
    # px = (X − cx)pz /fx, py = (Y − cy )pz /y
    
    width, height = depth_map.shape
    fx,fy,cx,cy,width_,height_ = camera_intrinsics

    point_cloud = []    
    for i in range(width):
        for j in range(height):
            pz = depth_map[i][j]
            px = ((i - cx)*pz)/fx
            py = ((j - cy)*pz)/fy
            point_cloud.append((px,py,pz))
                
    return point_cloud

def depth_to_pointcloud_fromlist(depth_map, camera_intrinsics):
    # generate point cloud from depth_map
    # px = (X − cx)pz /fx, py = (Y − cy )pz /y
    fx,fy,cx,cy,width_,height_ = camera_intrinsics

    point_cloud = []    
    for i,j,dz in depth_map:
         pz = dz
         px = ((i - cx)*pz)/fx
         py = ((j - cy)*pz)/fy
         point_cloud.append((px,py,pz))
                
    return point_cloud

def plot_pointcloud(point_cloud, title):
    import matplotlib
    matplotlib.use('Agg')  # Use a non-interactive backend
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [point[0] for point in point_cloud]
    ys = [point[1] for point in point_cloud]
    zs = [point[2] for point in point_cloud]
    ax.scatter(xs, ys, zs)
    # plt.show()
    plt.savefig(title + '.png')
    
def resample_depth_map(depth_map, object_pixels,width,height):
    # this method resamples the object depth map identified by object_pixels width width and height
    pixel_y , pixel_x = map(list, zip(*object_pixels))
    max_x = max(pixel_x)
    max_y = max(pixel_y)
    min_x = min(pixel_x)
    min_y = min(pixel_y)
    
    
    new_pixel_x = np.linspace(min_x,max_x, width).astype(int)
    new_pixel_y = np.linspace(min_y,max_y, height).astype(int)

    resampled_depth_map = []

    for x in new_pixel_x:
        for y in new_pixel_y:
            #if not math.isnan(depth_map[y,x]):
            resampled_depth_map.append((y,x,depth_map[y,x])) 
            
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

# Initialization of intrinsic and extrinsic parameters
focal_length_x = 610  # Focal length in pixels (along X-axis)
focal_length_y = 610  # Focal length in pixels (along Y-axis)
principal_point_x = 317  # Principal point offset along X-axis (in pixels)
principal_point_y = 238  # Principal point offset along Y-axis (in pixels)

image_height = 480
image_width = 640
image_dimensions = (image_height, image_width)

# Example of extracting intrinsic and extrinsic parameters
K = np.array([[focal_length_x, 0, principal_point_x],
                                [0, focal_length_y, principal_point_y],
                                [0, 0, 1]])

# Load file
object_name = "banana"
file_name = "cad_models/banana.obj"  
mesh_scale_real = 0.01 #0.01 banana
max_virtual_depth = 5 #[m]
mesh_scale = mesh_scale_real


# initialize nvisii
camera_intrinsic = [focal_length_x,focal_length_y,principal_point_x,principal_point_y,image_width,image_height]
interactive = False
initialize_nvisii(interactive, camera_intrinsic,object_name, file_name)




#  Euler angles rapresentation
euler_angles = [0,0,0] # radians - roll pitch and yaw
quaternion = euler_to_quaternion(euler_angles)#[0,0.5,0.5,0]  

# axis-angle rapresentation
#quaternion = axis_angle_to_quaternion([0,0,1],1.57)

#quaternion = normalize_quaternion(quaternion)
translation = np.array([0,0,0.6])



# Generate the depth map
depth_map, object_pixels = generate_depth_map(object_name,translation, quaternion) # The first time call it two times due to nvisii bug
depth_map, object_pixels = generate_depth_map(object_name,translation, quaternion)
cv2.imshow("depth_map", depth_map)


# # crop object image
obj_depth_image = crop_object_image(depth_map,object_pixels)
#cv2.imshow("obj_depth_image", obj_depth_image)


# # normalize object depth map
obj_depth_image_normalized = normalize_depth_map(obj_depth_image)
#cv2.imshow("obj_depth_image_normal", obj_depth_image)


# change object mesh to simulate differences between real object and cad
new_object_name = "banana2"
new_object_path = file_name#"cad_models/bowl.obj"
mesh_scale = mesh_scale
change_object_mesh(object_name, new_object_name, new_object_path)
translation2 = translation#compute_object_center(object_pixels, 1, camera_intrinsic)


mesh_scale_cad = mesh_scale_real*0.5
mesh_scale = mesh_scale_cad # change mesh scale to test different scales

initial_guess = [0,0,0] # euler angles
bnds = [(-3.14, 3.14), (-3.14, 3.14), (-3.14,3.14)]

#initial_guess = normalize_quaternion([1,0,0,0]) # quaternion
# constraints = {'type': 'eq', 'fun': unit_quaternion_constraint}

# initial_guess = [0.1,0.1,0.1,0]#0.09950372,0.001,0.99503719,1.00498756] # axis angle
# bnds = ((0.0001, 1), (0.0001, 1), (0.0001, 1), (0, 1))

# initial_guess = [0,1.57,1.57] # vector r'*theta - r' and theta are the axis and angle of rotation
# module_constraint = NonlinearConstraint(lambda x: np.linalg.norm(x), 0, 2*np.pi)


# result = minimize(orientation_cost_function,initial_guess,method='SLSQP',bounds=bnds,
#                   options={'ftol': 1e-2, 'eps': 1e-1,'maxiter': 10,'disp': True})
#result = minimize(orientation_cost_function,initial_guess,method='Powell',bounds=bnds, tol=1e-2,
#                 options={'ftol': 1e-2, 'eps': 10,'maxiter': 10,'disp': True})#, constraints=module_constraint)

# result = differential_evolution(orientation_cost_function, bounds=bnds, disp=True, workers=1, maxiter=10)


# result = dual_annealing(orientation_cost_function, bounds=[(-10, 10), (-10, 10), (-10, 10)])
# result = least_squares(orientation_cost_function, initial_guess, bounds=([-100, -100, -100], [100, 100, 100]))
# result = basinhopping(orientation_cost_function, initial_guess, niter=10, T=1.0, stepsize=0.5, minimizer_kwargs=None, take_step=None, accept_test=None, callback=None, interval=50, disp=True, niter_success=None, seed=None)   
# result = differential_evolution(orientation_cost_function, bounds=[(-10, 10), (-10, 10), (-10, 10)], disp=True)

# # second pose
#translation2 = [0.0,0.0,0.5]
orientation2 = [0,0,0]

#orientation2 = result.x
quaternion2 = euler_to_quaternion(orientation2)

# quaternion2 = axis_angle_to_quaternion(orientation2[0:3],orientation2[3])


# theta = np.linalg.norm(orientation2)
# axis = orientation2/theta
# orientation2 = np.concatenate((axis, [theta]))
# quaternion2 = axis_angle_to_quaternion(orientation2[0:3],orientation2[3])

quaternion2 = normalize_quaternion(quaternion2)
depth_map2, object_pixels2 = generate_depth_map(object_name,translation2, quaternion2)
obj_depth_image2 = crop_object_image(depth_map2,object_pixels2)
obj_depth_image2_normalized = normalize_depth_map(obj_depth_image2)    
cv2.imshow("depth_map2", depth_map2)


# Retrieve objects point clouds and resample them
resized_image_real_object, resized_image_cad_model = resize_images_to_same_size(obj_depth_image, obj_depth_image2)

cv2.imshow("resized_image_real_object", resized_image_real_object)
cv2.imshow("resized_image_cad_model", resized_image_cad_model)

res_width, res_height = resized_image_real_object.shape
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

# rotate point cloud in the camera frame first
nvisii_R_camera = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
# point_cloud_real = nvisii_R_camera.T @ point_cloud_real.T
# point_cloud_cad = nvisii_R_camera.T @ point_cloud_cad.T
# point_cloud_real = point_cloud_real.T
# point_cloud_cad = point_cloud_cad.T


plot_pointcloud(point_cloud_real,"point_cloud_real")
plot_pointcloud(point_cloud_cad,"point_cloud_cad")

#### brief test lemma umeyama rotation
# R = find_rotation_matrix(point_cloud_real, point_cloud_cad)
# R2 = quaternion_to_rotation_matrix(quaternion2)

# axis, angle = rotation_matrix_to_axis_angle(R)
# q_test = axis_angle_to_quaternion(axis, angle)
# q_test = normalize_quaternion(q_test)
# depth_test, object_pixels_test = generate_depth_map(object_name,translation2, q_test)
# cv2.imshow("depth_test", depth_test)
# print("axis", axis)
# print("angle", angle)
# print("R", R)


####

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
# plt.show()
plt.savefig("Result" + '.png')

#####################################################
# Compute real object estimated pose wrt camera frame 

R2 = quaternion_to_rotation_matrix(quaternion2)

cam_T_cad = np.eye(4)
cam_T_cad[:3, :3] = mesh_scale_cad*R2
cam_T_cad[:3, 3] = translation2


real_T_cad = np.eye(4)
real_T_cad[:3, :3] = c*R
real_T_cad[:3, 3] = t

cam_T_real = cam_T_cad @ np.linalg.inv(real_T_cad)

estimated_p_real = cam_T_real[:3, 3]
estimated_R_real = normalize_rotation_matrix(mesh_scale_cad*c*cam_T_real[:3, :3])
estimated_scale_real = mesh_scale_cad*c


print("real object position", estimated_p_real)
print("real object orientation", estimated_R_real) 
print("real object scale", estimated_scale_real)


cv2.waitKey(0)
cv2.destroyAllWindows()
nvisii.deinitialize()

