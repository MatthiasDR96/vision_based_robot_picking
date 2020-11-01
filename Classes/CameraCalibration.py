import glob
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Chessboard pattern
pattern = (7, 7)


# Gives all image path names for images in the file with 'filename'
def get_image_path_names(filename):
    return glob.glob(filename)


# Get a numpy array from file 'filename'
def get_numpy_data(filename):
    return np.load(filename, allow_pickle=True)


# Save a numpy array 'data' to a file 'filename'
def save_to_numpy(filename, data):
    return np.save(filename, data)


# Compute the mean transform from a list of transforms
def compute_mean_transformation(transformations):
    return np.mean(transformations, axis=0)


# Convert an RGB image to grayscale
def image_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Read an RGB image from a file 'filename'
def read_image(filename):
    return cv2.imread(filename)


# Shows an image
def show_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(100)


# Shows the image with the detected corners of the chessboard
def show_image_with_chessboard_corners(image, corners):
    img = cv2.drawChessboardCorners(image, pattern, corners, True)
    cv2.imshow('Calibration image', img)
    cv2.waitKey(100)


# Load the intrinsic camera matrix and distortion coefficients in a dictionary format form 'filename'
def load_intrinsic_camera_matrix(filename):
    intrinsic_camera_matrix = get_numpy_data(filename)
    data = intrinsic_camera_matrix['data'][()]
    mtx = data['MTX']
    dist = data['DIST']
    return mtx, dist


# Multiply two transformation_matrices A and B
def multiply_transforms(trans_A, trans_B):
    return np.dot(trans_A, trans_B)


# Invert a transformation matrix
def invert_transform(transform):
    return np.linalg.inv(transform)


# Find the pixel coordinates of the corners of the chessboard in 'image'
def find_corners(image):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    found, corners = cv2.findChessboardCorners(image, pattern, None)
    corners2 = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    return corners2


# Gives user defined object points for 1 chessboard
def get_object_points():
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * 0.02  # 20x20 mm
    return objp


# Get the intrinsic camera matrix using the point correspondences 'objpoints', and 'imgpoints'
def calibrate_camera(objpoints, imgpoints):

    # Read image to get the image shape
    img = cv2.imread('/content/drive/My Drive/object_pose_estimation_online/data/camera_calibration_images/CC_image_original_1.jpg')

    # To grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Intrinsic matrix
    camera_properties = dict(MTX=mtx, DIST=dist, RVECS=rvecs, TVECS=tvecs)

    # Print matrix
    print('\nMTX (Intrinsic camera matrix): \n')
    print(camera_properties['MTX'])
    print('\nDIST (Distortion coefficients): \n')
    print(camera_properties['DIST'])

    # Save matrix in numpy format
    np.savez("/content/drive/My Drive/object_pose_estimation_online/data/matrix_files/intrinsic_camera_properties.npz", data=camera_properties)
    return camera_properties


# Calculate the reprojection error of the intrinsic matrix
def calculate_reprojection_error(intrinsic_camera_matrix, objpoints, imgpoints):
    if not intrinsic_camera_matrix:
        print("No camera matrix found.")
        exit(0)
    else:
        mtx = intrinsic_camera_matrix['MTX']
        dist = intrinsic_camera_matrix['DIST']
        rvecs = intrinsic_camera_matrix['RVECS']
        tvecs = intrinsic_camera_matrix['TVECS']

        tot_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error
        return tot_error


# Get the extrinsic camera calibration using the point correspondences and intrinsic camera matrix
def extrinsic_calibration(frame, objp, corners, mtx, dist):
    # Extrinsic camera calibration
    retval, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)

    # Rotation_vector into rotation_matrix
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    # World origin in camera frame
    ct_transform = data_to_transform(rvec_matrix, translation_vector)

    # Camera's origin in world frame
    # camera_rotation_matrix = rvec_matrix.T
    # camera_translation_vector = -np.dot(rvec_matrix.T, translation_vector)
    # cam_world_transform = data_to_transform(camera_rotation_matrix, camera_translation_vector)

    # Draw and display lines and text on the image
    draw_show_on_image(frame, rotation_vector, translation_vector, mtx, dist)

    return ct_transform


# Convert euler angles to a rotation matrix
def euler_to_matrix(theta):
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]
    ])
    R_y = np.array([
        [math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]
    ])
    R_z = np.array([
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]
    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# Draw the axis of the chessboard frame on the image given world coordinates of the axis
def draw_show_on_image(frame, rotation_vector, translation_vector, mtx, dist):
    # Axis coordinates in world coordinates system
    axis = np.float32([[0.06, 0, 0], [0, 0.06, 0], [0, 0, 0.06], [0, 0, 0]])

    # Project 3D points to image plane
    axi_imgpts, jacobian = cv2.projectPoints(axis, rotation_vector, translation_vector, mtx, dist)

    cv2.line(frame, tuple(axi_imgpts[3].ravel()), tuple(axi_imgpts[1].ravel()), (0, 255, 0), 2)  # GREEN Y
    cv2.line(frame, tuple(axi_imgpts[3][0]), tuple(axi_imgpts[2].ravel()), (255, 0, 0), 2)  # BLUE Z
    cv2.line(frame, tuple(axi_imgpts[3, 0]), tuple(axi_imgpts[0].ravel()), (0, 0, 255), 2)  # RED x
    text_pos = (axi_imgpts[0].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(frame, 'X', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    text_pos = (axi_imgpts[1].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(frame, 'Y', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    text_pos = (axi_imgpts[2].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(frame, 'Z', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    text_pos = (axi_imgpts[3].ravel() + np.array([200, 50])).astype(int)
    # cv2.putText(frame, '1unit=2cm', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

    # Display the resulting frame
    plt.imshow(frame)

# Convert a rotation matrix and translation matrix into a transformation matrix
def data_to_transform(r_matrix, t_position):
    mat = np.hstack((r_matrix, t_position))
    mat = np.vstack((mat, [0.0, 0.0, 0.0, 1.0]))
    return mat
