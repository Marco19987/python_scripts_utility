import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler
import cv2
import image_helper as image_helper
import nvisii_helper as nvisii_helper
import geometric_helper as geometric_helper


# Step 1: Load the dataset
viewpoint_file = 'banana_viewpoints_cont_repr_8.pkl'   
with open(viewpoint_file, 'rb') as file:
    data = pickle.load(file)

images = np.array([element['depth_map'] for element in data])

for image in images:
    # substitue nan elements in image with 1000000
    image[np.isnan(image)] = 0

# convert images in grayscale
# cv2.imshow("First image", images[0])


orientations = np.array([element['continuos_representation'] for element in data])
# each orientation is a 3x2 matrix, we need to flatten it
orientations = orientations.reshape(orientations.shape[0], -1)


# sort images random and pick the first P images
indices = np.random.permutation(len(images))
train_images = images[indices]
sorted_orientations = orientations[indices]

test_samples = 500
train_orientations = sorted_orientations[test_samples:]
test_orientations = sorted_orientations[:test_samples]

test_images = train_images[:test_samples]
train_images = train_images[test_samples:]


# Assuming data is a list of images, where each image is a 2D numpy array
# Step 2: Flatten the images
num_train_images = len(train_images)
train_images_shape = train_images[0].shape  # Assuming all images have the same shape
print("image_shape", train_images_shape)
flattened_data = np.array([image.flatten() for image in train_images])
print("flattened_data", flattened_data.shape)

flattened_data_test = np.array([image.flatten() for image in test_images])


# Optional: Standardize the data before applying PCA
# scaler = StandardScaler()
# flattened_data_standardized = scaler.fit_transform(flattened_data)

# Step 3: Apply PCA
# Choose the number of components, e.g., 100, or set n_components between 0 and 1 to select the number of components for the desired explained variance.
pca = PCA(n_components=1200)
reduced_data = pca.fit_transform(flattened_data)
print("reduced_data", reduced_data.shape)


# plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# show the first N components of the PCA
import image_helper
print("pca components shape", pca.components_.shape)
N = 0
for i in range(N):
    component = pca.components_[i]
    component = component.reshape(train_images_shape)
    print("Component " + str(i), component.shape)
    print("max", np.max(component))
    print("min", np.min(component))
    cv2.imshow("Component " + str(i), image_helper.normalize_depth_map(component))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Now, reduced_data contains the dimensionality reduced dataset
# and reconstructed_images contains the reconstructed images after dimensionality reduction (for visualization)
# Optional: Inverse transform to reconstruct images
reconstructed_data = pca.inverse_transform(reduced_data)
reconstructed_images = reconstructed_data.reshape((num_train_images,) + train_images_shape)

for i in range(0):
    cv2.imshow("Reconstructed image " + str(i), reconstructed_images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test_reconstruction = pca.transform(flattened_data_test)
reconstructed_test_images = pca.inverse_transform(test_reconstruction)
for i in range(0):
    cv2.imshow("Reconstructed test image " + str(i), reconstructed_test_images[i].reshape(train_images_shape))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # or any other suitable model
from sklearn.metrics import mean_squared_error

# Assuming 'sorted_orientations' is correctly aligned with your 'train_images'
# Split your reduced data and orientations into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(reduced_data, train_orientations, test_size=0.4, random_state=42)

# Train a model
model = RandomForestRegressor(n_estimators=200,criterion="squared_error",random_state=42, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

# Predict orientations for the test set
predicted_orientations = model.predict(X_test)

# Evaluate the model (optional)
mse = mean_squared_error(y_test, predicted_orientations)
print(f"Mean Squared Error: {mse}")

# Use the model to predict orientations of new images
# First, you need to process these new images the same way you did with the training images
# For example:
# new_images_flattened = np.array([image.flatten() for image in new_images])
# new_images_standardized = scaler.transform(new_images_flattened)  # Use the same scaler as before
# new_images_reduced = pca.transform(new_images_standardized)  # Use the same PCA as before
# new_predicted_orientations = model.predict(new_images_reduced)

# test the predictor



# Initialization of intrinsic and extrinsic parameters
focal_length_x = 610.0  # Focal length in pixels (along X-axis)
focal_length_y = 610.0  # Focal length in pixels (along Y-axis)
principal_point_x = 317.0  # Principal point offset along X-axis (in pixels)
principal_point_y = 238.0  # Principal point offset along Y-axis (in pixels)
image_height = 200
image_width = 200
camera_intrinsics = [focal_length_x,focal_length_y,principal_point_x,principal_point_y,image_width,image_height]


# Load file real object
object_name = "banana"
file_name = "cad_models/banana.obj"  
mesh_scale = 0.01 #0.01 banana


max_virtual_depth = 5 #[m]


# Pose object
translation = np.array([-0.3,-0.2,1]) # position of the object in meters wrt camera
euler_angles = [0,0,0] # radians - roll pitch and yaw
quaternion_real = geometric_helper.euler_to_quaternion(euler_angles)

# initialize nvisii
interactive = False
nvisii_helper.initialize_nvisii(interactive, camera_intrinsics,object_name, file_name)


# Generate the real depth map
depth_map, object_pixels = image_helper.generate_depth_map(object_name,translation, quaternion_real, mesh_scale, camera_intrinsics, max_virtual_depth) # The first time call it two times due to nvisii bug



predicted_test_orientations = model.predict(test_reconstruction)
for i,prediction in enumerate(predicted_test_orientations):
    print("Predicted orientation", prediction)
    print("Real orientation", test_orientations[i])
    mse = mean_squared_error(test_orientations[i].T, prediction.T)

    # evaluate mse by hand
    print(f"Mean Squared Error: {mse}")
    cv2.imshow("Test image", test_images[i])
    cv2.imshow("Reconstructed test image", reconstructed_test_images[i].reshape(train_images_shape))
    R = geometric_helper.continuos_representation(prediction.reshape(3,2))
    quaternion_predicted = geometric_helper.rotation_matrix_to_quaternion(R)
    depth_map, object_pixels = image_helper.generate_depth_map(object_name,translation, quaternion_predicted, mesh_scale, camera_intrinsics, max_virtual_depth) # The first time call it two times due to nvisii bug
    cv2.imshow("Predicted depth map", image_helper.normalize_depth_map(depth_map))
    cv2.waitKey(0)


nvisii_helper.deinitialize_nvisii()

