import numpy as np

def read_obj_file(file_path):
    vertices = []
    faces = []
    normals = []
    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                # Leggi i vertici
                vertex = [float(coord) for coord in line.strip().split()[1:]]
                vertices.append(vertex)
            elif line.startswith('vn '):
                # Leggi le normali
                normal = [float(coord) for coord in line.strip().split()[1:]]
                normals.append(normal)
            elif line.startswith('f '):
                # Leggi le facce
                face_vertex = []
                face_normals = []
                for index in line.strip().split()[1:]:
                    vertex_index, normal_index = index.split('//')
                    face_vertex.append(int(vertex_index) - 1)
                    face_normals.append(int(normal_index) - 1)
                faces.append((face_vertex, face_normals))
                
    return vertices, faces, normals

def backface_culling(vertices, faces, normals, camera_position):
    visible_faces = []

    for i, face_indices in enumerate(faces):

        face_vertices = [vertices[idx] for idx in face_indices[:][0]]


        # # Calcola la normale della faccia
        # face_normals = [normals[idx] for idx in face_indices] 
        # print(face_normals)
        # #face_normal = normals[i]

        # # Calcola il vettore dalla fotocamera al centro della faccia
        # face_center = np.mean(face_vertices, axis=0)
        # camera_to_face = face_center - camera_position

        # # Verifica se la normale della faccia è rivolta verso la fotocamera
        # if np.dot(face_normal, camera_to_face) < 0:
        #     visible_faces.append(face_indices)
    return visible_faces

file_path = "cube.obj"
camera_position = np.array([0, 0, 10])  # Assumiamo che la camera sia situata all'origine con asse z positivo

# Definisci i parametri intrinseci ed estrinseci della fotocamera
focal_length_x = 1000  
focal_length_y = 1000  
principal_point_x = 320  
principal_point_y = 240  

rotation_00 = 1.0  
rotation_01 = 0.0  
rotation_02 = 0.0  
rotation_10 = 0.0  
rotation_11 = 1.0  
rotation_12 = 0.0  
rotation_20 = 0.0  
rotation_21 = 0.0  
rotation_22 = 1.0  

translation_x = 0.0  
translation_y = 0.0  
translation_z = 4  

image_height = 640
image_width = 480

# Costruisci le matrici intrinseche ed estrinseche
intrinsic_matrix = np.array([[focal_length_x, 0, principal_point_x],
                              [0, focal_length_y, principal_point_y],
                              [0, 0, 1]])

extrinsic_matrix = np.array([[rotation_00, rotation_01, rotation_02, translation_x],
                              [rotation_10, rotation_11, rotation_12, translation_y],
                              [rotation_20, rotation_21, rotation_22, translation_z]])

# Leggi il file .obj
vertices, faces, normals = read_obj_file(file_path)


# Esegui il Back-face Culling utilizzando i parametri intrinseci ed estrinseci della fotocamera
visible_faces = backface_culling(vertices, faces, normals, camera_position)

# Stampa le facce visibili dopo il Back-face Culling
print("Facce visibili dopo il Back-face Culling:", visible_faces)
