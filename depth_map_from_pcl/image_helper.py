import numpy as np
import matplotlib.pyplot as plt
import nvisii
import math
import cv2


def convert_from_uvd(h, w, d, fx, fy, cx, cy):
    px = (w - cx)/fx
    py = (h - cy)/fy
    
    z = d/np.sqrt(1. + px**2 + py**2)     
    return z

def generate_depth_map(object_name, position, quaternion, mesh_scale, camera_instrinsics, max_virtual_depth):
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

    rotation = nvisii.quat(qw,qx,qy,qz)
    #rotation = nvisii.angleAxis(qz,nvisii.vec3(qw,qx,qy))

    rotation_flip = nvisii.angleAxis(-nvisii.pi(),nvisii.vec3(1,0,0)) * rotation # additional rotation due camera nvissi frame
    obj_mesh.get_transform().set_rotation(rotation_flip)
    
    obj_mesh.get_transform().set_scale(nvisii.vec3(mesh_scale))

    focal_length_x, focal_length_y, principal_point_x, principal_point_y, image_width, image_height = camera_instrinsics
        
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
            
        
        resized_image_1 = cv2.resize(image1_array,(new_width, new_height), interpolation=cv2.INTER_AREA)
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
            
                    
        resized_image_1 = cv2.resize(image2_array,(new_width,new_height), interpolation=cv2.INTER_AREA) #INTER_AREA
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


def normalize_depth_map(depth_map):
    
    max_value = np.nanmax(depth_map)
    min_value = np.nanmin(depth_map)
    
    range_value = max_value - min_value
    
    # Normalize the matrix
    normalized_depth_map = (depth_map - min_value)/ range_value
    
    return normalized_depth_map
