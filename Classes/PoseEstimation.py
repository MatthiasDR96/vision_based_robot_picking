import copy
import glob

import numpy as np
import open3d as o3d
import pcl

voxel_size = 0.01
max_distance = 0.005


# Gives all image path names for images in the file with 'filename'
def get_image_path_names(filename):
    return glob.glob(filename)


# Get a numpy array from file 'filename'
def get_numpy_data(filename):
    return np.load(filename, allow_pickle=True)


# Save a numpy array 'data' to a file 'filename'
def save_to_numpy(filename, data):
    return np.save(filename, data)


# Read a pointcloud from a file 'filename'
def read_pointcloud(filename):
    return o3d.read_point_cloud(filename)


# Save the pointcloud to a file 'filename'
def save_pointcloud(pointcloud, filename):
    pcl.save(pointcloud, filename)


# Visualize pointcloud
def show_pointcloud(pointcloud):
    o3d.io.draw_geometries([pointcloud])


# Multiply two transformation matrices A and B
def multiply_transforms(trans_A, trans_B):
    return np.dot(trans_A, trans_B)


# Transform the pointcloud into another coordinate frame using the transformation matrix
def transform_pointcloud(pointcloud, transform):
    pointcloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pointcloud.transform(transform)
    return pointcloud


# Covert pointcloud from open3d format to pcl format for plane segmentation
def pcd_to_pcl_format(pointcloud):
    o3d.io.write_point_cloud('data/pointcloud_files/tmp.pcd', pointcloud)
    return pcl.io.load('data/pointcloud_files/tmp.pcd')


# Convert the transformation matrix to an understandable pose
def transformation_to_pose(transformation_matrix):
    trans = (transformation_matrix[0][3], transformation_matrix[1][3], transformation_matrix[2][3])
    return trans


# Filter the pointcloud
def filter_pointcloud(pointcloud):
    return o3d.geometry.voxel_down_sample(pointcloud, voxel_size)


# Crop the pointcloud (you can play with the bounds)
def crop_pointcloud(pointcloud):
    pointcloud_ = np.asanyarray(pointcloud.points)
    mask_x1 = pointcloud_[:, 0] < 0.3
    mask_x2 = pointcloud_[:, 0] > -0.3
    mask_y1 = pointcloud_[:, 1] > -1.0
    mask_y2 = pointcloud_[:, 1] < -0.5
    cropped = pointcloud_[np.multiply(np.multiply(mask_x1, mask_x2), np.multiply(mask_y1, mask_y2))]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cropped)
    return pcd


# Segment the table from other objects, inliers = table pointcloud, outliers = objects pointcloud
def ransac_plane_segmentation(point_cloud, max_distance=max_distance):
    segmenter = point_cloud.make_segmenter()
    segmenter.set_model_type(pcl.SACMODEL_PLANE)
    segmenter.set_method_type(pcl.SAC_RANSAC)
    segmenter.set_distance_threshold(max_distance)
    inlier_indices, coefficients = segmenter.segment()
    inliers = point_cloud.extract(inlier_indices, negative=False)
    outliers = point_cloud.extract(inlier_indices, negative=True)
    return inliers, outliers


# Read in source and target pointcloud and preprocess
def prepare_dataset(source, target):
    print(":: Load two point clouds.")
    draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source)
    target_down, target_fpfh = preprocess_point_cloud(target)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


# Preprocess pointcloud before registration
def preprocess_point_cloud(pcd):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    o3d.geometry.estimate_normals(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd_down,
                                                     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                                                                          max_nn=100))
    return pcd_down, pcd_fpfh


# Do global registration on target and source pointcloud (RANSAC)
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    draw_registration_result(source_down, target_down, result.transformation)
    return result


# Do local registration on target and source pointcloud (ICP)
def refine_registration(source, target, source_fpfh, target_fpfh, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    draw_registration_result(source, target, result.transformation)
    return result


# Visualize the registration result using two pointclouds and the obtained transformation
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
