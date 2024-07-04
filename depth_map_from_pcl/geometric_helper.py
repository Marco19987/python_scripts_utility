import numpy as np

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





def read_obj_file(file_name):
    import trimesh
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
    
def axis_angle_viewpoint(phi, theta, psi):
    # method to generate the corresponding orientatin in the compact
    # axis angle representations from the three angles phi, theta, psi
    # phi is the angle of the versor in the xy plane
    # theta is the angle of the versor with the z axis
    # psi is the angle of rotation around the versor
    rx = np.cos(theta) * np.cos(phi) 
    ry = np.cos(theta) * np.sin(phi)
    rz = np.sin(theta)
    rpsi = np.array([rx, ry, rz]) * psi
    return rpsi

def axis_angle_from_vector(rtheta):
    # from the vector r*theta extract the axis r and the angle theta
    theta = np.linalg.norm(rtheta)
    axis = rtheta/theta if theta != 0 else [0,0,1]
    return axis, theta

    

