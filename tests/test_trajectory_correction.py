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
    fig.draw_points(map_points, name='Map points')
    
    
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
    
    
    # # draw map camera poses
    # for id, pose in zip(map_cameras_id, map_cameras_pose):
    #     name = "est pose " + str(id)
    #     fig.draw_frame(pose, name=name)
    fig.draw_points(map_cameras_gt_tvec, name='tvec')
    
    
    # # draw ground truth camera poses
    # for id, pose in zip(map_cameras_gt_id, map_cameras_gt_pose):
    #     name = "gt pose " + str(id)
    #     fig.draw_frame(pose, name=name)
    fig.draw_points(map_cameras_gt_tvec, name='gt_tvec')
    
       
    S_guess= np.eye(4)
    points = map_cameras_gt_tvec
    measurements = map_cameras_tvec
    n_iterations = 500
    
    Similarity, chi_stats, errors_sicp = robust_sicp(S_guess, points, measurements, n_iterations, damping=0.5, kernel_threshold=0.0001)
    _scale, _rotation, _translation = Similarity[3, 3], Similarity[:3, :3], Similarity[:3, 3]
    
    print("Similarity")
    print(Similarity)
    
    print(_scale)
    print(_rotation)
    print(_translation)
    print()
    
    errors_sicp_figure = Drawer()
    errors_sicp_figure.plot_error(errors_sicp)
    
    # correct the measurements (map pose) using the similarity
    map_cameras_tvec_sim_correction = np.array([similarity_transform(np.linalg.inv(Similarity), p) for p in measurements])
    fig.draw_points(map_cameras_tvec_sim_correction, color=np.array([[255], [0], [0]]), name = "sim_correction_tvec")
        
    X_guess = np.eye(4)
    points = map_cameras_gt_tvec
    measurements = map_cameras_tvec_sim_correction
    n_iterations = 50
    Affine, chi_stats, errors_micp = robust_micp(X_guess, points, measurements, n_iterations, damping=0.5, kernel_threshold=0.1)
    _scale, _rotation, _translation = Affine[3, 3], Affine[:3, :3], Affine[:3, 3]
    
    
    print("Affine")
    print(Affine)
    
    errors_micp_figure = Drawer()
    errors_micp_figure.plot_error(errors_micp)
    
    # correct the measurements (map pose) using the affinity
    # oss: the measuremnts are already adjusted using the similarity
    map_cameras_gt_tvec_affine_correction = np.array([transform(np.linalg.inv(Affine), p) for p in map_cameras_tvec_sim_correction])
    fig.draw_points(map_cameras_gt_tvec_affine_correction, color=np.array([[255], [0], [0]]), name = "aff_correction_gt_tvec")
    
    fig.show()
    


if __name__ == "__main__":
    main()