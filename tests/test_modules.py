import sys
sys.path.append('/home/leonardo/elective_project/')
# sys.path.append('/home/leonardo/elective_project/simulation')


from segmentator import *
from least_squares import *
from simulation.environment import *
from utils.geometric_utils import *
from utils.vision_utils import *
from utils.reconstruction_utils import *

from drawer import *
from dataset import DatasetType, SfMDataset


import argparse
import numpy as np
import open3d as o3d

import pycolmap

from hloc.utils.read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec


        


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COLMAP binary and text models")
    parser.add_argument("--input_model", required=True, help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    
    # load dataset
    workdir = "/mnt/d"
    data = SfMDataset(workdir, DatasetType.ICRA)
    
    # load reconstruction
    reconstruction = pycolmap.Reconstruction(args.input_model)
    map_points_keys = sorted(reconstruction.points3D)
    map_cameras_keys = sorted(reconstruction.cameras)
    map_images_keys = sorted(reconstruction.images)
    
    # ESTIMATED POSE
    tvecs = np.array([reconstruction.images[key].tvec for key in map_images_keys])
    # tvecs = np.array([tvec @ np.diag([-1, 1, 1]) for tvec in tvecs])
    # tvecs = np.array([tvec @ np.diag([1, -1, 1]) for tvec in tvecs])
    # tvecs = np.array([tvec @ np.diag([1, 1, -1]) for tvec in tvecs])
        
    
    
    # reconstruction image filenames
    fns = np.array([reconstruction.images[key].name for key in map_images_keys])
    
    # load data ground truth poses
    data.load_poses()
    data.load_image_data(fns)
    
    # GROUND TRUCTH POSE
    gt_tvecs = np.array([data.gt_images[key].tvec for key in data.gt_images])
    
    
    # load data images
    data.load_image_data(fns)
    
    # Load tracker
    tracker = SegmentationModule()
    
    
    drawer = Drawer()
    # drawer.draw_init(tracker.classes)
    
    segmention_window = DrawerWindow(WindowName.Segmentation, WindowType.Plot2D) 
    segmention_window.draw_init(tracker.classes)   
    drawer.add_window(segmention_window)
    
    pose_window = DrawerWindow(WindowName.Poses, WindowType.Plot3D)
    drawer.add_window(pose_window)
    
    points_window = DrawerWindow(WindowName.Points, WindowType.Plot3D)
    drawer.add_window(points_window)    
    
    
    for image_key in map_images_keys:
        drawer.clear_windows()
        
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
        
        segments = tracker.segmentation(rgb_img)
        # segments = tracker.get_valid_segments(segments)
        keypoints_segmentation_mask, points = tracker.get_points_inside_segments(segments, points2d_coords)
        keypoints_segmentation_mask_idx = np.array([id for id, m in enumerate(keypoints_segmentation_mask) if m])
        
        total_keypoints = len(keypoints_segmentation_mask) # total keypoints found in the current image
        total_segmented_keypoints = np.sum(keypoints_segmentation_mask) # total keypoints segmentated in the current image
        # TODO a 'segmented keypoint' is a keypoint (u, v) inside a valid segmented region
        
        window = drawer.windows[WindowName.Segmentation]
        window.draw_image(rgb_img) 
        window.draw_segments(rgb_img, segments)
        window.draw_keypoints(points, keypoints_segmentation_mask)
        
        # keypoints outside the region segmented will not be used for optimization
        # TODO: It simulates the RFID  (we obtain depth informations only by Beacons)
        points2d_coords = points2d_coords[keypoints_segmentation_mask_idx]
        
        
        # 3d point in the world frame
        depth_map = data.gt_images_depth[image_key]

        # get sensor reading for the 2d keypoints extracted (they are 3d points in the world frame)
        points3d_gt_coords = np.array([depth_map[v, u].xyz for u, v in points2d_coords])
        
        # if 1 the sensor reading is valid
        # 0 otherwise
        depth_sensor_mask = np.array([depth_map[v, u].error for u, v in points2d_coords])
        depth_sensor_mask_idx = np.array([id for id, m in enumerate(depth_sensor_mask) if m])
        
        # 3d points of the map
        points3d_coords = points3d_coords[depth_sensor_mask_idx]
        # 3d points of the ground trutch (sensor readings)
        points3d_gt_coords = points3d_gt_coords[depth_sensor_mask_idx] 
        
        

        
        S_guess = np.eye(4) # initial guess
        n_iterations = 500
        
        Similarity, chi_stats, num_inliers = robust_sicp(X_guess=S_guess, 
                                                         points=points3d_coords, 
                                                         measurements=points3d_gt_coords, 
                                                         n_iterations=n_iterations, damping=0.5, kernel_threshold=0.05)
        _scale, _rotation, _translation = Similarity[3, 3], Similarity[:3, :3], Similarity[:3, 3]
        print("Similarity")
        print(Similarity)
        
        # drawer.plot_chi_stats(chi_stats)
        # drawer.plot_num_inliers(num_inliers)
        
        # correct the map points using the similarity (scale, rotation and translation)
        points3d_coords_similarity_correction = np.array([similarity_transform(Similarity, p) for p in points3d_coords])
        similarity_error = np.linalg.norm(points3d_gt_coords-points3d_coords_similarity_correction, axis=1)
        # print(similarity_error)
        
        # correct the map poses using the similarity (scale, rotation and translation)
        tvecs_similarity_correction = np.array([similarity_transform(Similarity, tvec) for tvec in tvecs])
        
        
        X_guess = np.eye(4) # intial guess
        n_iterations = 30
        
        Affinity, chi_stats, num_inliers = robust_micp(X_guess = X_guess, 
                                                     points=points3d_coords_similarity_correction, 
                                                     measurements=points3d_gt_coords, 
                                                     n_iterations=n_iterations, damping=0.7, kernel_threshold=0.05)
        _scale, _rotation, _translation = Affinity[3, 3], Affinity[:3, :3], Affinity[:3, 3]
        print("Affinity")
        print(Affinity)
        # drawer.plot_chi_stats(chi_stats)
        # drawer.plot_num_inliers(num_inliers)
        
        # correct the map points using the affinity (rotation and translation)
        points3d_coords_affinity_correction = np.array([transform(Affinity, p) for p in points3d_coords_similarity_correction])
        affinity_error = np.linalg.norm(points3d_gt_coords-points3d_coords_affinity_correction, axis=1)
        # print(affinity_error)
        
        # correct the map poses using the affinity (rotation and translation)
        tvecs_affinity_correction = np.array([transform(Affinity, p) for p in tvecs_similarity_correction])
        
        # plotting
        window = drawer.windows[WindowName.Points]
        window.plot_points(points3d_coords, 'r') # est
        window.plot_points(points3d_gt_coords, 'b') # gt
        window.plot_points(points3d_coords_similarity_correction, 'g')
        window.plot_points(points3d_coords_affinity_correction, 'g')
        
        
        window = drawer.windows[WindowName.Poses]
        window.plot_tvecs(tvecs, 'r') # est
        window.plot_tvecs(gt_tvecs, 'b') # gt
        window.plot_tvecs(tvecs_similarity_correction, 'g')
        window.plot_tvecs(tvecs_affinity_correction, 'g')
        
        
        drawer.update_windows()
        
        input("Press Enter to continue...")
        
        
        
        
        
        
        
        
        
        
        
        
    # fig.show()
    
    
    
    


if __name__ == "__main__":
    main()