import argparse
import numpy as np
import open3d as o3d

import pycolmap

from hloc.utils import viz_3d
from hloc.utils.read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec

from dataset import DatasetType, SfMDataset
from utils.vision_utils import homogeneous_to_rotation_translation, rototranslation_to_homogeneous

import imageio.v3 as iio
from utils.vision_utils import *
from drawer import *
        


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COLMAP binary and text models")
    parser.add_argument("--input_model", required=True, help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    
    workdir = "/mnt/d"
    
    
    
    # TODO see pycolmap: https://github.com/colmap/pycolmap/blob/master/reconstruction/reconstruction.cc
    # for colmap python bindings 
    reconstruction = pycolmap.Reconstruction(args.input_model)
    
    map_points_keys = sorted(reconstruction.points3D)
    map_cameras_keys = sorted(reconstruction.cameras)
    map_images_keys = sorted(reconstruction.images)
    
    print("map points keys")
    print(map_points_keys)
    print("map cameras keys")
    print(map_cameras_keys)
    print("map images keys")
    print(map_images_keys)
    
    # load
    fns = np.array([reconstruction.images[key].name for key in map_images_keys])
    # this avoids to process the entire dataset if not strictly needed
    data = SfMDataset(workdir, DatasetType.ICRA)
    data.load_poses()
    data.load_image_data(fns)
    
    
    fig = BrowserDrawer()
    
    map_points = np.array([reconstruction.points3D[key].xyz for key in map_points_keys])
    map_colors = np.array([reconstruction.points3D[key].color for key in map_points_keys])
    print("Map points shape:", map_points.shape)
    
    # draw 3D map points
    fig.draw_points(map_points, map_colors)
    
    
    map_cameras_tvec = np.array([reconstruction.images[key].tvec for key in map_images_keys])
    map_cameras_qvec = np.array([reconstruction.images[key].qvec for key in map_images_keys])
    map_cameras_id = np.array([reconstruction.images[key].image_id for key in map_images_keys])
    map_cameras_pose = np.array([np.hstack([qvec2rotmat(qvec), tvec.reshape(-1, 1)]) for (tvec, qvec) in zip(map_cameras_tvec, map_cameras_qvec)])
    
    print("Map pose shape:", map_cameras_pose.shape)
    
    map_cameras_gt_tvec = np.array([data.gt_images[key].tvec for key in data.gt_images])
    map_cameras_gt_qvec = np.array([data.gt_images[key].qvec for key in data.gt_images])
    map_cameras_gt_id = np.array([data.gt_images[key].image_id for key in data.gt_images])
    map_cameras_gt_pose = np.array([np.hstack([qvec2rotmat(qvec), tvec.reshape(-1, 1)]) for (tvec, qvec) in zip(map_cameras_gt_tvec, map_cameras_gt_qvec)])
    
    print("Gt pose shape:", map_cameras_gt_pose.shape)
    
    
    
    # draw map camera poses
    for id, pose in zip(map_cameras_id, map_cameras_pose):
        name = "est pose " + str(id)
        fig.draw_frame(pose, name=name)
    
    # draw ground truth camera poses
    for id, pose in zip(map_cameras_gt_id, map_cameras_gt_pose):
        name = "gt pose " + str(id)
        fig.draw_frame(pose, name=name)
        
    # scales__ = np.array([tvec_/tvec_gt_ for tvec_, tvec_gt_ in zip(map_cameras_tvec, map_cameras_gt_tvec)])
    # print(scales__)
    # exit()
    
    # fig.draw_lines(map_cameras_gt_tvec, map_cameras_tvec)
    # fig.show()
    
    # # compute intial guess
    # _scale, _rotation, _translation = absolute_scale_estimation_closed_form(map_cameras_tvec, map_cameras_gt_tvec)
    # S_guess = np.zeros((4, 4))
    # S_guess[:3, :3] = _rotation
    # S_guess[:3, 3] = _translation
    # S_guess[3, 3] = _scale
    # print("Similarity Initial guess")
    # print(S_guess)
    
    # X_guess = S_guess
    
    X_guess= np.eye(4)
    points = map_cameras_gt_tvec
    measurements = map_cameras_tvec
    n_iterations = 100
    
    X, chi_stats = robust_sicp(X_guess, points, measurements, n_iterations, damping=0.5, kernel_threshold=10)
    _scale, _rotation, _translation = X[3, 3], X[:3, :3], X[:3, 3]
    print("Similarity")
    print(X)
    
    # X_inv = np.linalg.inv(X)
    # _scale = X_inv[3, 3]
    
    map_cameras_pose_recovered = np.array([np.hstack([_rotation.T @ pose[:3, :3], (_rotation.T @ pose[:3, 3] + _translation).reshape(-1, 1)])/_scale  for pose in map_cameras_pose])
    # map_cameras_pose_recovered = np.array([(X.T @ rototranslation_to_homogeneous(pose))/_scale for pose in map_cameras_pose])
    # map_cameras_pose_recovered = np.array([(X_inv @ rototranslation_to_homogeneous(pose))/_scale for pose in map_cameras_pose])
    # map_cameras_pose_recovered = np.array([np.hstack([_rotation @ pose[:3, :3], (_scale * _rotation.T @ pose[:3, 3] + _translation).reshape(-1, 1)]) for pose in map_cameras_pose])
    # map_cameras_pose_recovered = np.array([np.hstack([pose[:3, :3], (_scale * pose[:3, 3] + _translation).reshape(-1, 1)]) for pose in map_cameras_pose])
    print(map_cameras_pose_recovered.shape)
    
    for id, pose in zip(map_cameras_id, map_cameras_pose_recovered):
        name = "recovered pose " + str(id)
        fig.draw_frame(pose, name=name, color=[0, 0, 0])
    
    
    
    Sims = []
    
    intrinsic = data.calibration_matrix
    fx, fy, px, py = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    
    for image_key in map_images_keys:
        
        
        image = reconstruction.images[image_key]
        
        points3d_indices = np.array([point2d.point3D_id for point2d in image.get_valid_points2D()]) 
        
        points2d_coords = np.array([[int(point2d.xy[0]), int(point2d.xy[1])] for point2d in image.get_valid_points2D()]) 
        points3d_coords = np.array([reconstruction.points3D[id].xyz for id in points3d_indices])
        

        data.load_depth_data(image.name)
        depth_map = data.gt_images_depth[image_key]
        points3d_gt_coords = np.array([depth_map[v, u].xyz for u, v in points2d_coords])
        
        mask = np.array([depth_map[v, u].error for u, v in points2d_coords])
        mask_idx = np.array([id for id, m in enumerate(mask) if m])
        
        print(points3d_gt_coords.shape, points3d_coords.shape)
        points3d_coords = points3d_coords[mask_idx]
        points3d_gt_coords = points3d_gt_coords[mask_idx] 
        print(points3d_gt_coords.shape, points3d_coords.shape)
        
        X_guess, points, measurements, n_iterations = np.eye(4), points3d_gt_coords, points3d_coords, 1000
        X, chi_stats = robust_sicp(X_guess, points, measurements, n_iterations, damping=1000, kernel_threshold=10000)
        print(chi_stats)
        print(X)
        Sims.append(X)
        
    exit()
        
    _scale, _rotation, _translation = Sims[-1][3, 3], Sims[-1][:3, :3], Sims[-1][:3, 3], 
    
    map_points_recovered = np.array([(_rotation.T @ reconstruction.points3D[key].xyz + _translation)/_scale for key in map_points_keys])
    
    name = "Scaled 3d points"
    fig.draw_points(map_points_recovered, map_colors, name=name)
    
    map_cameras_pose_recovered = np.array([np.hstack([_rotation.T @ pose[:3, :3], (_rotation.T @ pose[:3, 3] + _translation).reshape(-1, 1)])/_scale  for pose in map_cameras_pose])
    # map_cameras_pose_recovered = np.array([(X.T @ rototranslation_to_homogeneous(pose))/_scale for pose in map_cameras_pose])
    # map_cameras_pose_recovered = np.array([(X_inv @ rototranslation_to_homogeneous(pose))/_scale for pose in map_cameras_pose])
    # map_cameras_pose_recovered = np.array([np.hstack([_rotation @ pose[:3, :3], (_scale * _rotation.T @ pose[:3, 3] + _translation).reshape(-1, 1)]) for pose in map_cameras_pose])
    # map_cameras_pose_recovered = np.array([np.hstack([pose[:3, :3], (_scale * pose[:3, 3] + _translation).reshape(-1, 1)]) for pose in map_cameras_pose])
    print(map_cameras_pose_recovered.shape)
    
    for id, pose in zip(map_cameras_id, map_cameras_pose_recovered):
        name = "recovered pose " + str(id)
        fig.draw_frame(pose, name=name, color=[0, 0, 0])
    
    fig.show()
    
    
    
    


if __name__ == "__main__":
    main()