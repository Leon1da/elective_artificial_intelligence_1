import sys
sys.path.append('/home/leonardo/elective_artificial_intelligence_1/')

from simulation.environment import *
from utils.geometric_utils import *
from utils.vision_utils import *
from drawer import *
from dataset import DatasetType, SfMDataset


import argparse
import numpy as np
import open3d as o3d

import pycolmap

        


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COLMAP binary and text models")
    parser.add_argument("--input_model", required=True, help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    
    
    # TODO see pycolmap: https://github.com/colmap/pycolmap/blob/master/reconstruction/reconstruction.cc
    # for colmap python bindings 
    reconstruction = pycolmap.Reconstruction(args.input_model)
    
    map_points_keys = sorted(reconstruction.points3D)
    map_cameras_keys = sorted(reconstruction.cameras)
    map_images_keys = sorted(reconstruction.images)
    
    # print("map points keys")
    # print(map_points_keys)
    # print("map cameras keys")
    # print(map_cameras_keys)
    # print("map images keys")
    # print(map_images_keys)
    
    fig = BrowserDrawer()
    
    map_points = np.array([reconstruction.points3D[key].xyz for key in map_points_keys])
    print("Map points shape:", map_points.shape)
    
    # draw 3D map points
    fig.draw_points(map_points, name='Map points')
    
    
    map_cameras_tvec = np.array([reconstruction.images[key].tvec for key in map_images_keys])
    map_cameras_qvec = np.array([reconstruction.images[key].qvec for key in map_images_keys])
    map_cameras_id = np.array([reconstruction.images[key].image_id for key in map_images_keys])
    
    print("Map pose shape:", map_cameras_tvec.shape)
    
    # draw 3D map poses
    fig.draw_points(map_cameras_tvec, name='Map poses')
    
    
    fig.show('test_sfm_visualization')



if __name__ == "__main__":
    main()