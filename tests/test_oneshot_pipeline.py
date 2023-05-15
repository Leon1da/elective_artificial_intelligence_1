import sys
sys.path.append('/home/leonardo/elective_artificial_intelligence_1/')



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
    args = parser.parse_args()
    return args


def main():
    
    
    args = parse_args()

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
    
    # load dataset
    workdir = "/mnt/d"
    data = SfMDataset(workdir, DatasetType.ICRA)
    
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
    
    
    # Init windows (Visualization)
    drawer = Drawer()
    
    segmention_window = SegmentationWindow(WindowName.Segmentation, classes=image_segmentator_module.classes) 
    drawer.add_window(segmention_window)
    
    posecomplete_window = PosesWindow(WindowName.PosesComplete)
    drawer.add_window(posecomplete_window)
    
    poseevaluation_window = PosesWindow(WindowName.PosesEvaluation)
    drawer.add_window(poseevaluation_window)
    
    points_window = PointsWindow(WindowName.Points)
    drawer.add_window(points_window)
    
    # scale_statistics_window = ScaleEstimationWindow(WindowName.ScaleStatistics)
    # drawer.add_window(scale_statistics_window)    
    
    
    for image_key in map_images_keys:
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
        keypoints_segmentation_mask, points = image_segmentator_module.get_keypoints_inside_segments(segments, points2d_coords)
        keypoints_segmentation_mask_idx = np.array([id for id, m in enumerate(keypoints_segmentation_mask) if m])
        
        total_keypoints = len(keypoints_segmentation_mask) # total keypoints found in the current image
        total_segmented_keypoints = np.sum(keypoints_segmentation_mask) # total keypoints segmentated in the current image
        # TODO a 'segmented keypoint' is a keypoint (u, v) inside a valid segmented region
        
        print('Found', total_keypoints, 'inside the image.')
        print('Found', total_segmented_keypoints, 'inside the segments.')
        
        if not total_segmented_keypoints: 
            print('Skipping scale correction. No data available.')
            continue
        
        
        drawer.clear(window_name=WindowName.Segmentation)
        drawer.draw(window_name=WindowName.Segmentation, image=filename_rgb, segments=segments, keypoints=points, mask=keypoints_segmentation_mask)
        
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
        # 3d points of the ground trutch (sensor readings)
        points3d_gt_coords = points3d_gt_coords[depth_sensor_mask_idx] 
        
        print('Computing initial guess..')
        print('Run Closed-form Absolute scale estimation..')
        s_, R_, t_ = absolute_scale_estimation_closed_form(points3d_coords,points3d_gt_coords)
        # print(s_, R_, t_)
        print('OK.')
        print('scale:', s_)
        
        scale_estimator_module.similarity_transformation_guess = srt_to_similarity(s_, R_, t_)
        
        # Recovering
        iterations = 1000
        dumping = 0.6
        kernel_threshold = 0.001
        scale_estimator_module.configure(iterations=iterations, dumping=dumping, kernel_threshold=kernel_threshold, verbose=False)
        print('Run Similarity Iterative Closest Point..')
        print('##### Configuration #####')
        print('### Iteration:', iterations)
        print('### Dumping:', dumping)
        print('### Kernel threshold:', kernel_threshold)
        print('### Keypoints number:', total_segmented_keypoints)
        print('[START] opimization..')
        Similarity, chi_evolution, num_inliers_evolution, similarity_evolution = scale_estimator_module.recover_similarity(points=points3d_coords, measurements=points3d_gt_coords)
        print('[END] ok.')    
        s_, R_, t_ = similarity_to_srt(Similarity)
        print("Similarity")
        print(Similarity)
        
        # correct the map points using the similarity (scale, rotation and translation)
        points3d_coords_similarity_correction = np.array([similarity_transform(Similarity, p) for p in points3d_coords])
        
        # correct the map poses using the similarity (scale, rotation and translation)
        tvecs_similarity_correction = np.array([similarity_transform(Similarity, tvec) for tvec in tvecs])
        rotmat_similarity_correction = np.array([similarity_transform(Similarity, rot) for rot in rotmat])
        
        
        scale_estimator_module.linear_transformation_guess = rotation_translation_to_homogeneous(R_, t_)
        
        iterations = 500
        dumping = 0.6
        kernel_threshold = 0.005
        scale_estimator_module.configure(iterations=iterations, dumping=dumping, kernel_threshold=kernel_threshold, verbose=False)

        print('Run Rigid Iterative Closest Point..')
        print('##### Configuration #####')
        print('### Iteration:', iterations)
        print('### Dumping:', dumping)
        print('### Kernel threshold:', kernel_threshold)
        print('### Keypoints number:', total_segmented_keypoints)
        print('[START] opimization..')
        Transform, chi_evolution, num_inliers_evolution, transform_evolution = scale_estimator_module.recover_linear_transformation(points=points3d_coords_similarity_correction, measurements=points3d_gt_coords)
        print('[END] ok.')
        print('Transform')
        print(Transform)
        
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
        
        traj_length = gt_trajectory_length[-1]
        ATE_percentage = relative_position_error(ate_mean_rigid, traj_length)
        print('Percentage Translation Error / Trajectory Length:', ATE_percentage, '/', traj_length)
        
            
        
        # plots 
        # 1) translation error (percentage) / trajectory length 
        # 2) translation error (meters) / number icp iterations
        # 3) translation error (meters) / number keyframes
        # 3) translation error (meters) / number keypoints (we mean keypoints used for optimization)
        
        
        drawer.clear(window_name=WindowName.PosesComplete)
        drawer.draw(window_name=WindowName.PosesComplete, tvecs=tvec_normalized, color='#ff0000')
        drawer.draw(window_name=WindowName.PosesComplete, tvecs=gt_tvec_normalized, color='#0000ff')
        drawer.draw(window_name=WindowName.PosesComplete, tvecs=tvecs_similarity_correction_normalized, color='#00ffff')
        drawer.draw(window_name=WindowName.PosesComplete, tvecs=tvecs_homogeneous_correction_normalized, color='#00ff00')
        
        drawer.clear(window_name=WindowName.PosesEvaluation)
        drawer.draw(window_name=WindowName.PosesEvaluation, tvecs=gt_tvec_normalized, color='#0000ff')
        drawer.draw(window_name=WindowName.PosesEvaluation, tvecs=tvecs_homogeneous_correction_normalized, color='#00ff00')
        
        
        
        
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
        break
    
    
    input("Press Enter to print results on the browser.")
    
    fig = BrowserDrawer()
    
    # draw 3D map pose
    fig.draw_points(tvecs, color=np.array([[255], [0], [0]]), name='Estimation')
    fig.draw_points(gt_tvecs, color=np.array([[0], [0], [255]]), name='Ground Truth')
    fig.draw_points(tvecs_similarity_correction, color=np.array([[0], [255], [255]]), name='Similarity')
    fig.draw_points(tvecs_homogeneous_correction, color=np.array([[0], [255], [0]]), name='Rigid')
    
    fig.show('test_oneshot_pipeline')
    
            
        
        

if __name__ == "__main__":
    main()