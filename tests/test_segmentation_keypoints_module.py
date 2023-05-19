import sys
sys.path.append('/home/leonardo/elective_artificial_intelligence_1/')

from dataset import DatasetType, SfMDataset
from segmentator import SegmentationModule

from utils.reconstruction_utils import *

from drawer import Drawer, ScaleEstimationWindow, SegmentationWindow, WindowName

from PIL import Image
import argparse
import numpy as np

import pycolmap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COLMAP binary and text models")
    parser.add_argument("--input_model", required=True, help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")
    args = parser.parse_args()
    return args

def plot_keypoints_histogram(fig, ax, keypoints):
    
    keyframes, total_keypoints, segmentated_keypoints, optimization_keypoints = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
    
    bins = [keyframes, keyframes + 1]
    
    x = [total_keypoints]
    ax.stairs(x, bins, lw=1, ec="#000000", fc="#29339B", alpha=1, fill=True)
    
    x = [segmentated_keypoints]
    ax.stairs(x, bins, lw=1, ec="#000000", fc="#348AA7", alpha=1, fill=True)

    x = [optimization_keypoints]            
    ax.stairs(x, bins, lw=1, ec="#000000", fc="#5DD39E", alpha=1, fill=True)
    
    ax.set_xlabel('# keyframe', fontsize="20")
    ax.set_ylabel('# keypoints', fontsize="20")
    
    c1 = mpatches.Patch(color='#29339B', label='total keypoints')
    c2 = mpatches.Patch(color='#348AA7', label='segmented keypoints')
    c3 = mpatches.Patch(color='#5DD39E', label='otimization keypoints')
    
    
    ax.legend(handles=[c1, c2, c3], loc="upper right", fontsize="20")
    
    fig.canvas.draw_idle() # draw on the same image
    
    

def main():
        
    args = parse_args()
    
    workdir = "/mnt/d"
    
    data = SfMDataset(workdir, DatasetType.ICRA)
    # data.load_poses()
    
    # load reconstruction
    reconstruction = pycolmap.Reconstruction(args.input_model)
    map_points_keys = sorted(reconstruction.points3D)
    map_cameras_keys = sorted(reconstruction.cameras)
    map_images_keys = sorted(reconstruction.images)
    
    # reconstruction image filenames
    fns = np.array([reconstruction.images[key].name for key in map_images_keys])
    
    # load data ground truth poses
    data.load_poses()
    # load data images
    data.load_image_data(fns)
    
    
    image_segmentator_module = SegmentationModule()
    
    # Init windows (Visualization)
    drawer = Drawer()
    fig, ax = plt.subplots(1, 1)
        
    
    segmention_window = SegmentationWindow(WindowName.Segmentation, classes=image_segmentator_module.classes) 
    drawer.add_window(segmention_window)
    
    for index, image_key in enumerate(map_images_keys):
        
        # get reconstruction image by key
        image = reconstruction.images[image_key]
        
        # get image points that have been mapped as 3d points
        points2d = image.get_valid_points2D()
        points2d_coords = get_coordinates(points2d)
        
        # load rgb image
        filename_rgb = data.load_image(image.name)
        
        print("Process:")
        print(" - ", filename_rgb)
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
        
        # Random Sampling 
        min_sampling_num = 50
        max_sampling_num = 300
        sampling_percentage = 0.25 # 10 %
        sampling_num = int(total_segmented_keypoints * sampling_percentage)
        if sampling_num < min_sampling_num: 
            sampling_num = min_sampling_num
        elif sampling_num > max_sampling_num:
            sampling_num = max_sampling_num
        if total_segmented_keypoints < sampling_num:
            sampling_num = total_segmented_keypoints
            
    
        drawer.clear(window_name=WindowName.Segmentation)
        
        drawer.draw(window_name=WindowName.Segmentation, image=filename_rgb, segments=segments)
        
        plot_keypoints_histogram(fig, ax, [index, total_keypoints, total_segmented_keypoints, sampling_num])
        
        drawer.update()
        
        
        


if __name__ == "__main__":
    
   input('Press enter to start.')
   main()
   input('Press enter to terminate')