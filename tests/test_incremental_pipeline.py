import sys, os
sys.path.append(os.getcwd())


from utils.dataset_utils import *
from utils.geometric_utils import *
from utils.vision_utils import *
from utils.reconstruction_utils import *

from evaluation import *
        
from drawer import *
from dataset import DatasetType, SfMDataset
from segmentator import *
from scale_estimator import ScaleEstimatorModule

import argparse
import numpy as np

import pycolmap

import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COLMAP binary and text models")
    parser.add_argument("--input_model", required=True, help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")
    parser.add_argument("--dataset_path", help="path to dataset folder", default="")
    args = parser.parse_args()
    return args


def main():
    
    
    args = parse_args()
    
    # load dataset
    workdir = os.getcwd() if args.dataset_path == "" else args.dataset_path
    data = SfMDataset(workdir, DatasetType.ICRA)
    
    # load reconstruction
    reconstruction = pycolmap.Reconstruction(args.input_model)
    map_points_keys = sorted(reconstruction.points3D)
    map_cameras_keys = sorted(reconstruction.cameras)
    map_images_keys = sorted(reconstruction.images)
    
    # ESTIMATED POSE
    tvecs = np.array([reconstruction.images[key].tvec for key in map_images_keys])
    rotmat = np.array([quaternion_to_rotation(reconstruction.images[key].qvec) for key in map_images_keys])
    
    # reconstruction image filenames
    fns = np.array([reconstruction.images[key].name for key in map_images_keys])
    
    # load data ground truth poses
    data.load_poses()
    # load data images
    data.load_image_data(fns)
    
    # GROUND TRUTH POSE
    gt_tvecs = np.array([data.gt_images[key].tvec for key in data.gt_images])
    gt_rotmat = np.array([quaternion_to_rotation(data.gt_images[key].qvec) for key in data.gt_images])
    
    gt_trajectory_length = compute_total_trajectory_displacement(gt_tvecs)    
    
    # print(np.divide(tvecs, gt_tvecs))
    
    # Load Image Segmentator
    image_segmentator_module = SegmentationModule()
    
    # Load Scale Estimator
    scale_estimator_module = ScaleEstimatorModule()
    
    total_scale_estimator_iterations = 0
    
    # Init windows (Visualization)
    drawer = Drawer()
    
    segmention_window = SegmentationWindow(WindowName.Segmentation, classes=image_segmentator_module.classes) 
    drawer.add_window(segmention_window)
    
    drawer.add_window(PosesWindow(WindowName.PosesComplete))
    
    drawer.add_window(PosesWindow(WindowName.PosesEvaluation))
    
    drawer.add_window(PosesWindow3D(WindowName.PosesComplete3d))
    
    points_window = PointsWindow(WindowName.Points)
    drawer.add_window(points_window)
    
    scale_statistics_window = ScaleEstimationWindow(WindowName.ScaleStatistics)
    drawer.add_window(scale_statistics_window)    
    
    global_points = np.zeros(shape=(1, 3))
    global_points_gt = np.zeros(shape=(1, 3))
    global_points_sim_corrected = np.zeros(shape=(1, 3))
    
    for index, image_key in enumerate(map_images_keys):
        # get reconstruction image by key
        image = reconstruction.images[image_key]
        
        # get image points that have been mapped as 3d points
        points2d = image.get_valid_points2D()
        points2d_coords = get_coordinates(points2d)
        
        # get corresponding 3d points
        points3d_indices = get_point3D_ids(points2d)
        points3d_coords = np.array([reconstruction.points3D[id].xyz for id in points3d_indices])
        points3d_coords = np.array([point3d_coords @ np.diag([-1, 1, 1]) for point3d_coords in points3d_coords])
        # points3d_coords = np.array([point3d_coords @ np.diag([1, -1, 1]) for point3d_coords in points3d_coords])
        # points3d_coords = np.array([point3d_coords @ np.diag([1, 1, -1]) for point3d_coords in points3d_coords])
        
        # load rgb image
        filename_rgb = data.load_image(image.name)
        # load the sensor readings for this image (the depth information)
        filename_rgbd = data.load_depth_data(image.name)
        
        
        print("Process:")
        print(" - ", filename_rgb)
        print(" - ", filename_rgbd)
        rgb_img = Image.open(filename_rgb)
        
        segments = image_segmentator_module.segmentation(rgb_img)
        # segments = tracker.get_valid_segments(segments)
        keypoints_segmentation_mask, keypoints = image_segmentator_module.get_keypoints_inside_segments(segments, points2d_coords)
        keypoints_segmentation_mask_idx = np.array([id for id, m in enumerate(keypoints_segmentation_mask) if m])
        
        total_keypoints = len(keypoints_segmentation_mask) # total keypoints found in the current image
        total_segmented_keypoints = np.sum(keypoints_segmentation_mask) # total keypoints segmentated in the current image
        # TODO a 'segmented keypoint' is a keypoint (u, v) inside a valid segmented region
        
        print('Found', total_keypoints, 'inside the image.')
        print('Found', total_segmented_keypoints, 'inside the segments.')
        
        
        drawer.clear(window_name=WindowName.Segmentation)
        drawer.draw(window_name=WindowName.Segmentation, image=filename_rgb, segments=segments, keypoints=keypoints, mask=keypoints_segmentation_mask)
        
        if not total_segmented_keypoints: 
            print('Skipping scale correction. No data available.')
            drawer.update()
            continue
        
        
        # keypoints outside the region segmented will not be used for optimization
        # TODO: It simulates the RFID  (we obtain depth informations only from Beacons)
        points2d_coords = points2d_coords[keypoints_segmentation_mask_idx]
        
        key = path_to_id(image.name)
        # 3d point in the world frame
        depth_map = data.gt_images_depth[key]

        # get sensor reading for the 2d keypoints extracted (they are 3d points in the world frame)
        points3d_gt_coords = np.array([depth_map[v, u].xyz for u, v in points2d_coords])
        
        # 1 if the sensor reading is valid
        # 0 otherwise
        depth_sensor_mask = np.array([depth_map[v, u].error for u, v in points2d_coords])
        depth_sensor_mask_idx = np.array([id for id, m in enumerate(depth_sensor_mask) if m])
        
        # 3d points of the map
        points3d_coords = points3d_coords[depth_sensor_mask_idx]
        # 3d points of the ground truth (sensor readings)
        points3d_gt_coords = points3d_gt_coords[depth_sensor_mask_idx] 
        
        
        global_points = np.vstack((global_points, points3d_coords))[1:, ]
        global_points_gt = np.vstack((global_points_gt, points3d_gt_coords))[1:, ]
        num_global_points = global_points.shape[0] - 1
        
        global_points = points3d_coords
        global_points_gt = points3d_gt_coords
        num_global_points = global_points.shape[0]
        
        # Random Sampling 
        min_sampling_num = 100
        max_sampling_num = 5000
        sampling_percentage = 0.25 # 10 %
        sampling_num = int(num_global_points * sampling_percentage)
        if sampling_num < min_sampling_num: 
            sampling_num = min_sampling_num
        elif sampling_num > max_sampling_num:
            sampling_num = max_sampling_num
        if num_global_points < sampling_num:
            sampling_num = num_global_points
            
        print('### Random Sampling')
        print('### sampling_num:', sampling_num)
            
        idx = np.arange(num_global_points)
        idx = np.random.choice(idx, sampling_num, replace=False)
        
        
        points = global_points[idx] 
        measurements =  global_points_gt[idx]
        
        
        # Recovering
        iterations = 100
        dumping = 0.6
        kernel_threshold = 0.005
        scale_estimator_module.configure(iterations=iterations, dumping=dumping, kernel_threshold=kernel_threshold, verbose=False)
        
        print('Run Rigid Iterative Closest Point..')
        print('##### Configuration #####')
        print('### Iteration:', iterations)
        print('### Dumping:', dumping)
        print('### Kernel threshold:', kernel_threshold)
        print('### points:', len(points))
        print('### measurements:', len(measurements))
        Similarity, chi_evolution, num_inliers_evolution, similarity_evolution = scale_estimator_module.recover_similarity_transformation(points=points, measurements=measurements)
        
        print("Similarity")
        print(Similarity)
        
        # correct the map points using the similarity (scale, rotation and translation)
        points3d_coords_similarity_correction = np.array([similarity_transform(Similarity, p) for p in points3d_coords])
        
        # correct the map poses using the similarity (scale, rotation and translation)
        tvecs_similarity_correction = np.array([similarity_transform(Similarity, tvec) for tvec in tvecs])
        rotmat_similarity_correction = np.array([similarity_transform(Similarity, rot) for rot in rotmat])
        
        
        # Similarity correction to the points applied in order to run the refinment using Ricp
        points = np.array([similarity_transform(Similarity, p) for p in points])
        
        iterations = 100
        dumping = 0.6
        kernel_threshold = 0.005
        scale_estimator_module.configure(iterations=iterations, dumping=dumping, kernel_threshold=kernel_threshold, verbose=False)
        
        print('Run Rigid Iterative Closest Point..')
        print('##### Configuration #####')
        print('### Iteration:', iterations)
        print('### Dumping:', dumping)
        print('### Kernel threshold:', kernel_threshold)
        print('### points:', len(points))
        print('### measurements:', len(measurements))
        Transform, chi_evolution, num_inliers_evolution, transform_evolution = scale_estimator_module.recover_rigid_transformation(points=points, measurements=measurements)
        
        
        print('Transform')
        print(Transform)
        
        total_scale_estimator_iterations = total_scale_estimator_iterations + iterations
        
        # correct the map points using the affinity (rotation and translation)
        points3d_coords_homogeneous_correction = np.array([homogeneous_transform(Transform, p) for p in points3d_coords_similarity_correction])
            
        # correct the map poses using the affinity (rotation and translation)
        tvecs_homogeneous_correction = np.array([homogeneous_transform(Transform, p) for p in tvecs_similarity_correction])
        rotmat_homogeneous_correction = np.array([homogeneous_transform(Transform, rot) for rot in rotmat_similarity_correction])
        
        
        
        # Normalization (We center all the trayectory in absolute zero)
        gt_tvec_normalized = np.array([t - gt_tvecs[0] for t in gt_tvecs]) # ground-truth
        tvec_normalized = np.array([t - tvecs[0] for t in tvecs]) # estimation
        tvecs_similarity_correction_normalized = np.array([t - tvecs_similarity_correction[0] for t in tvecs_similarity_correction]) # SICP
        tvecs_homogeneous_correction_normalized = np.array([t - tvecs_homogeneous_correction[0] for t in tvecs_homogeneous_correction]) # SICP + RICP
        
        # Compute the Error
        
        ate_mean_similarity, ate_std_similarity, ate_min_similarity = absolute_position_error(tvecs_similarity_correction_normalized, gt_tvec_normalized)
        print('Absolute Translation Error (SICP):')
        print('MEAN', ate_mean_similarity)
        print('STD', ate_std_similarity)
        print('MIN', ate_min_similarity)
        
        ate_mean_rigid, ate_std_rigid, ate_min_rigid = absolute_position_error(tvecs_homogeneous_correction_normalized, gt_tvec_normalized)
        print('Absolute Translation Error (RICP):')
        print('MEAN', ate_mean_rigid)
        print('STD', ate_std_rigid)
        print('MIN', ate_min_rigid)
        
        # compute the error percentage with respect to the trajectory lenght
        traj_length = gt_trajectory_length[index]
        if traj_length:
            ATE_percentage = relative_position_error(ate_mean_rigid, traj_length)
        else:
            ATE_percentage = 100
        print('Percentage Translation Error / Trajectory Length:', ATE_percentage, '/', traj_length)
        
        drawer.clear(window_name=WindowName.PosesComplete)
        drawer.draw(window_name=WindowName.PosesComplete, tvecs=tvec_normalized, color='#ff0000')
        drawer.draw(window_name=WindowName.PosesComplete, tvecs=gt_tvec_normalized, color='#0000ff')
        drawer.draw(window_name=WindowName.PosesComplete, tvecs=tvecs_similarity_correction_normalized, color='#00ffff')
        drawer.draw(window_name=WindowName.PosesComplete, tvecs=tvecs_homogeneous_correction_normalized, color='#00ff00')
        
        drawer.clear(window_name=WindowName.PosesComplete3d)
        drawer.draw(window_name=WindowName.PosesComplete3d, tvecs=tvec_normalized, color='#ff0000')
        drawer.draw(window_name=WindowName.PosesComplete3d, tvecs=gt_tvec_normalized, color='#0000ff')
        drawer.draw(window_name=WindowName.PosesComplete3d, tvecs=tvecs_similarity_correction_normalized, color='#00ffff')
        drawer.draw(window_name=WindowName.PosesComplete3d, tvecs=tvecs_homogeneous_correction_normalized, color='#00ff00')
        
        
        drawer.clear(window_name=WindowName.PosesEvaluation)
        drawer.draw(window_name=WindowName.PosesEvaluation, tvecs=gt_tvec_normalized, color='#0000ff')
        drawer.draw(window_name=WindowName.PosesEvaluation, tvecs=tvecs_homogeneous_correction_normalized, color='#00ff00')
        
        
        # plots 
        # 1) translation error (percentage) / trajectory length 
        # 2) translation error (meters) / number icp iterations
        # 3) translation error (meters) / number keyframes
        # 3) translation error (meters) / number keypoints (we mean keypoints used for optimization)
        drawer.draw(window_name=WindowName.ScaleStatistics, 
                    # absolute_error_iterations = [ATE_hom, total_scale_estimator_iterations],
                    # absolute_error_keypoints = [ATE_hom, num_global_points], 
                    absolute_error_keyframes = [ate_mean_rigid, index], 
                    percentage_error_trajectory_length = [ATE_percentage, traj_length],
                    scale_keyframes = [Similarity[3, 3], index], 
                    # keypoints = [index, total_keypoints, total_segmented_keypoints, sampling_num]
                    keypoints = [index, total_keypoints, total_segmented_keypoints, 0]
                    )
        
        
        drawer.clear(window_name=WindowName.Points)
        drawer.draw(window_name=WindowName.Points, points=points3d_coords, color='#ff0000')
        drawer.draw(window_name=WindowName.Points, points=points3d_gt_coords, color='#0000ff')
        drawer.draw(window_name=WindowName.Points, points=points3d_coords_similarity_correction, color='#00ffff')
        drawer.draw(window_name=WindowName.Points, points=points3d_coords_homogeneous_correction, color='#00ff00')
        
        drawer.update()
        
        # To avoid useless waste of memory space
        # we deallocate the sensor readings that accumulate over time
        del data.gt_images_depth[key]
        gc.collect()
        # input("Press Enter to proceed with the next iteration...")
        
        
    input("Press Enter to exit...")
            


if __name__ == "__main__":
    main()