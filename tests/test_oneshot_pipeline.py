import sys
sys.path.append('/home/leonardo/elective_artificial_intelligence_1/')



from utils.dataset_utils import path_to_id
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
    
    # print(np.divide(tvecs, gt_tvecs))
    
    # Load Image Segmentator
    image_segmentator_module = SegmentationModule()
    
    # Load Scale Estimator
    scale_estimator_module = ScaleEstimatorModule()
    
    
    # Init windows (Visualization)
    drawer = Drawer()
    
    segmention_window = SegmentationWindow(WindowName.Segmentation, classes=image_segmentator_module.classes) 
    drawer.add_window(segmention_window)
    
    pose_window = PosesWindow(WindowName.Poses)
    drawer.add_window(pose_window)
    
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
        keypoints_segmentation_mask, points = image_segmentator_module.get_points_inside_segments(segments, points2d_coords)
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
        
        
        s_, R_, t_ = absolute_scale_estimation_closed_form(points3d_coords,points3d_gt_coords)
        print(s_, R_, t_)
        input('Press enter to continue...')
        
        scale_estimator_module.similarity_transformation_guess = srt_to_similarity(s_, R_, t_)
        
        # Recovering
        iterations = 1000
        dumping = 0.6
        kernel_threshold = 0.001
        scale_estimator_module.configure(iterations=iterations, dumping=dumping, kernel_threshold=kernel_threshold, verbose=False)
        Similarity, chi_evolution, num_inliers_evolution, similarity_evolution = scale_estimator_module.recover_similarity(points=points3d_coords, measurements=points3d_gt_coords)
        s_, R_, t_ = similarity_to_srt(Similarity)
        print("Similarity")
        print(Similarity)
        
        # correct the map points using the similarity (scale, rotation and translation)
        points3d_coords_similarity_correction = np.array([similarity_transform(Similarity, p) for p in points3d_coords])
        
        # correct the map poses using the similarity (scale, rotation and translation)
        tvecs_similarity_correction = np.array([similarity_transform(Similarity, tvec) for tvec in tvecs])
        rotmat_similarity_correction = np.array([similarity_transform(Similarity, rot) for rot in rotmat])
        
        # Compute the Error
        ATE_sim = absolute_position_error(tvecs_similarity_correction, gt_tvecs)
        print('Absolute Translation Error (Similarity):', ATE_sim)
                
        # ARE_sim = absolute_rotation_error(rotmat_similarity_correction, gt_rotmat)
        # print('Absolute Rotation Error (Similarity):', ARE_sim)
        
        # drawer.draw(window_name=WindowName.ScaleStatistics, 
        #             sicp_transformation_evolution=similarity_evolution, 
        #             sicp_chi_evolution=list(chi_evolution), 
        #             sicp_inliers_evolution=list(num_inliers_evolution),
        #             sicp_error=similarity_error)
    
        scale_estimator_module.linear_transformation_guess = rotation_translation_to_homogeneous(R_, t_)
        
        iterations = 500
        dumping = 0.6
        kernel_threshold = 0.005
        scale_estimator_module.configure(iterations=iterations, dumping=dumping, kernel_threshold=kernel_threshold, verbose=False)
        Transform, chi_evolution, num_inliers_evolution, transform_evolution = scale_estimator_module.recover_linear_transformation(points=points3d_coords_similarity_correction, measurements=points3d_gt_coords)
        print('Transform')
        print(Transform)
        
        # correct the map points using the affinity (rotation and translation)
        points3d_coords_homogeneous_correction = np.array([homogeneous_transform(Transform, p) for p in points3d_coords_similarity_correction])
    
        # correct the map poses using the affinity (rotation and translation)
        tvecs_homogeneous_correction = np.array([homogeneous_transform(Transform, p) for p in tvecs_similarity_correction])
        rotmat_homogeneous_correction = np.array([homogeneous_transform(Transform, rot) for rot in rotmat_similarity_correction])
        
        # Compute the Error
        ATE_hom = absolute_position_error(tvecs_homogeneous_correction, gt_tvecs)
        print('Absolute Translation Error (Transform):', ATE_hom)
        
        # ARE_hom = absolute_rotation_error(rotmat_homogeneous_correction, gt_rotmat)
        # print('Absolute Rotation Error (Transform):', ARE_hom)
        
        # drawer.draw(window_name=WindowName.ScaleStatistics, 
        #             micp_transformation_evolution=transform_evolution, 
        #             micp_chi_evolution=list(chi_evolution), 
        #             micp_inliers_evolution=list(num_inliers_evolution),
        #             micp_error=linear_error)

        
        drawer.clear(window_name=WindowName.Points)
        drawer.draw(window_name=WindowName.Points, points=points3d_coords, color='#ff0000')
        drawer.draw(window_name=WindowName.Points, points=points3d_gt_coords, color='#0000ff')
        drawer.draw(window_name=WindowName.Points, points=points3d_coords_similarity_correction, color='#00ffff')
        drawer.draw(window_name=WindowName.Points, points=points3d_coords_homogeneous_correction, color='#00ff00')
        
        
        drawer.clear(window_name=WindowName.Poses)
        drawer.draw(window_name=WindowName.Poses, tvecs=tvecs, color='#ff0000')
        drawer.draw(window_name=WindowName.Poses, tvecs=gt_tvecs, color='#0000ff')
        drawer.draw(window_name=WindowName.Poses, tvecs=tvecs_similarity_correction, color='#00ffff')
        drawer.draw(window_name=WindowName.Poses, tvecs=tvecs_homogeneous_correction, color='#00ff00')
        
        
        drawer.update()
        
        # To avoid useless waste of memory space
        # we deallocate the sensor readings that accumulate over time
        del data.gt_images_depth[key]
        gc.collect()
        input("Press Enter to continue...")
        break
        
    input("Press Enter to exit...")
            
        
        
        
        
        
        
        
        
        
        
        
    # fig.show()
    
    
    
    


if __name__ == "__main__":
    main()