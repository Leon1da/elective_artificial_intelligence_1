import sys
sys.path.append('/home/leonardo/elective_project/')
# sys.path.append('/home/leonardo/elective_project/simulation')

from least_squares import *
from simulation.environment import *
from utils.geometric_utils import *
from utils.vision_utils import *
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
    # fig.draw_points(map_points, name='map points')
    
    # TODO the REFLECTION matrix is np.diag([-1, 1, 1])
    # TODO the reflection is needed beacuse the reconstruction is performed using a calibration matrix K 
    # with the fy element flippend from '-' to '+' beacuse of an implmentation detail.
    # this cause a reflection effect to the reconstruction, but since we assume that the calibration matrix is known
    # we simply reflect(mirror) the reconstruction once it has been computed, without lose of generality
    
    map_cameras_tvec = np.array([reconstruction.images[key].tvec for key in map_images_keys])
    # # TODO flip it!
    # map_cameras_tvec = np.array([tvec @ np.diag([-1, 1, 1])for tvec in map_cameras_tvec])
    
    map_cameras_qvec = np.array([reconstruction.images[key].qvec for key in map_images_keys])
    map_cameras_id = np.array([reconstruction.images[key].image_id for key in map_images_keys])
    map_cameras_pose = np.array([np.hstack([qvec2rotmat(qvec), tvec.reshape(-1, 1)]) for (tvec, qvec) in zip(map_cameras_tvec, map_cameras_qvec)])
    
    
    print("Map pose shape:", map_cameras_pose.shape)
    
    map_cameras_gt_tvec = np.array([data.gt_images[key].tvec for key in data.gt_images])
    map_cameras_gt_qvec = np.array([data.gt_images[key].qvec for key in data.gt_images])
    map_cameras_gt_id = np.array([data.gt_images[key].image_id for key in data.gt_images])
    map_cameras_gt_pose = np.array([np.hstack([qvec2rotmat(qvec), tvec.reshape(-1, 1)]) for (tvec, qvec) in zip(map_cameras_gt_tvec, map_cameras_gt_qvec)])
    
    
    print("Gt pose shape:", map_cameras_gt_pose.shape)
    
    print(map_cameras_tvec.shape)
    print(map_cameras_gt_tvec.shape)
    
    
    # # draw map camera poses
    # for id, pose in zip(map_cameras_id, map_cameras_pose):
    #     name = "est pose " + str(id)
    #     fig.draw_frame(pose, name=name)
    fig.draw_points(map_cameras_tvec, color=np.array([[255.], [0.], [0.]]), name='est tvec')
    
    
    
    # # draw ground truth camera poses
    # for id, pose in zip(map_cameras_gt_id, map_cameras_gt_pose):
    #     name = "gt pose " + str(id)
    #     fig.draw_frame(pose, name=name)
    fig.draw_points(map_cameras_gt_tvec, color=np.array([[0.], [0.], [255.]]), name='gt tvec')
    
    # fig.draw_lines(map_cameras_tvec, map_cameras_gt_tvec)
    
    Sims = []
    
    for image_key in map_images_keys:
        
        
        image = reconstruction.images[image_key]
        
        points3d_indices = np.array([point2d.point3D_id for point2d in image.get_valid_points2D()]) 
        
        points2d_coords = np.array([[int(point2d.xy[0]), int(point2d.xy[1])] for point2d in image.get_valid_points2D()]) 
        
        
        points3d_coords = np.array([reconstruction.points3D[id].xyz for id in points3d_indices])
        # Reflection of points (mirror effect to the reconstruction)
        points3d_coords = np.array([point3d_coords @ np.diag([-1, 1, 1]) for point3d_coords in points3d_coords])
        

        data.load_depth_data(image.name)
        depth_map = data.gt_images_depth[image_key]
        points3d_gt_coords = np.array([depth_map[v, u].xyz for u, v in points2d_coords])
        
        mask = np.array([depth_map[v, u].error for u, v in points2d_coords])
        mask_idx = np.array([id for id, m in enumerate(mask) if m])
        
        print(points3d_gt_coords.shape, points3d_coords.shape)
        points3d_coords = points3d_coords[mask_idx]
        points3d_gt_coords = points3d_gt_coords[mask_idx] 
        print(points3d_gt_coords.shape, points3d_coords.shape)
        
        # # Random sampling implmentation
        # num_sample = points3d_gt_coords.shape[0] # total number of sample
        
        # percentace = 0.1
        # num_to_sample = int(num_sample * percentace)
        # num_to_sample = 100
        # print("Sample #", num_to_sample)
        # print("percentage: ", percentace)
        # print("Point to sample #", num_to_sample)
        
        # idx = np.arange(num_sample)
        # idx_to_sample = np.random.choice(idx, num_to_sample, replace=False)
        
        # points3d_gt_coords = points3d_gt_coords[idx_to_sample]
        # points3d_coords = points3d_coords[idx_to_sample]
        
        
        fig.draw_points(points3d_gt_coords, color=np.array([[0.], [0.], [255.]]), name = "gt map")
        fig.draw_points(points3d_coords, color=np.array([[255.], [0.], [0.]]), name = "est map")
        # fig.draw_lines(points3d_gt_coords, points3d_coords, color=np.array([[125.], [125.], [125.]]), name = "correspondence")

        S_guess = np.eye(4)
        points = points3d_gt_coords
        measurements = points3d_coords
        n_iterations = 500
        
        Similarity, chi_stats, errors_sicp = robust_sicp(S_guess, points, measurements, n_iterations, damping=0.5, kernel_threshold=0.05)
        _scale, _rotation, _translation = Similarity[3, 3], Similarity[:3, :3], Similarity[:3, 3]
        
        print("Similarity")
        print(Similarity)
        
        print(_scale)
        print(_rotation)
        print(_translation)
        print()
        
        errors_sicp_figure = Drawer()
        errors_sicp_figure.plot_error(errors_sicp)
        
        # correct the measurements (map points) using the similarity
        measurements_similarity_correction = np.array([similarity_transform(np.linalg.inv(Similarity), p) for p in points3d_coords])
        # fig.draw_points(measurements_similarity_correction, color=np.array([[0.], [255.], [0.]]), name = "est map (similarity)")
        
        # correct the poses (map poses) using the similarity
        map_cameras_tvec_similarity_correction = np.array([similarity_transform(np.linalg.inv(Similarity), p) for p in map_cameras_tvec])
        # fig.draw_points(map_cameras_tvec_similarity_correction, color=np.array([[255.], [0], [0.]]), name = "est tvec (similarity)")

        
        X_guess = np.eye(4)
        
        points = points3d_gt_coords
        measurements = measurements_similarity_correction
        n_iterations = 30
        Affine, chi_stats, errors_micp = robust_micp(X_guess, points, measurements, n_iterations, damping=0.7, kernel_threshold=0.01)
        _scale, _rotation, _translation = Affine[3, 3], Affine[:3, :3], Affine[:3, 3]
        
        
        print("Affine")
        print(Affine)
        
        errors_micp_figure = Drawer()
        errors_micp_figure.plot_error(errors_micp)
        
        # further correct the measurements (map points) using the affinity
        measurements_affinity_correction = np.array([transform(np.linalg.inv(Affine), p) for p in measurements_similarity_correction])
        fig.draw_points(measurements_affinity_correction, color=np.array([[0.], [255.], [0.]]), name = "est map (affinity)")
        fig.draw_lines(measurements_affinity_correction, points3d_gt_coords, color=np.array([[128.], [128.], [128.]]), name = "point corrispondence")
    
        
        # further correct the poses (map poses) using the affinity
        map_cameras_tvec_affine_correction = np.array([transform(np.linalg.inv(Affine), p) for p in map_cameras_tvec_similarity_correction])
        # TODO flip it!
        # map_cameras_tvec_affine_correction = np.array([tvec @ np.diag([-1, 1, 1])for tvec in map_cameras_tvec_affine_correction])
    
        fig.draw_points(map_cameras_tvec_affine_correction, color=np.array([[0.], [255.], [0.]]), name = "est tvec (affinity)")
        fig.draw_lines(map_cameras_tvec_affine_correction, map_cameras_gt_tvec, color=np.array([[128.], [128.], [128.]]), name = "tvec corrispondence")
    
        
        
        fig.show()
        
        
        
        break
    
    fig.show()
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