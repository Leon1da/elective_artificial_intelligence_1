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
    
    
    # Load Scale Estimator
    scale_estimator_module = ScaleEstimatorModule()
    
    
    # Init windows (Visualization)
    drawer = Drawer()
    
    drawer.add_window(PosesWindow(WindowName.PosesComplete))
    
    drawer.add_window(PosesWindow(WindowName.PosesEvaluation))
    
    drawer.add_window(PosesWindow3D(WindowName.PosesComplete3d))

    
    # Recovering
    iterations = 5000
    dumping = 0.6
    kernel_threshold = 0.005
    scale_estimator_module.configure(iterations=iterations, dumping=dumping, kernel_threshold=kernel_threshold, verbose=False)
    print('Run Similarity Iterative Closest Point..')
    print('##### Configuration #####')
    print('### Iteration:', iterations)
    print('### Dumping:', dumping)
    print('### Kernel threshold:', kernel_threshold)

    Similarity, chi_evolution, num_inliers_evolution, similarity_evolution = scale_estimator_module.recover_similarity(points=tvecs, measurements=gt_tvecs)
    s_, R_, t_ = similarity_to_srt(Similarity)
    print('OK.')
    print("Similarity")
    print(Similarity)
    
    
    
    # correct the map poses using the similarity (scale, rotation and translation)
    tvecs_similarity_correction = np.array([similarity_transform(Similarity, tvec) for tvec in tvecs])
    rotmat_similarity_correction = np.array([similarity_transform(Similarity, rot) for rot in rotmat])
    
    # Compute the Error
    ATE_sim = absolute_position_error(tvecs_similarity_correction, gt_tvecs)
    print('Absolute Translation Error (Similarity):', ATE_sim)
            
   
    scale_estimator_module.linear_transformation_guess = rotation_translation_to_homogeneous(R_, t_)
    
    iterations = 1000
    dumping = 0.6
    kernel_threshold = 0.005
    scale_estimator_module.configure(iterations=iterations, dumping=dumping, kernel_threshold=kernel_threshold, verbose=False)

    print('Run Rigid Iterative Closest Point..')
    print('##### Configuration #####')
    print('### Iteration:', iterations)
    print('### Dumping:', dumping)
    print('### Kernel threshold:', kernel_threshold)

    Transform, chi_evolution, num_inliers_evolution, transform_evolution = scale_estimator_module.recover_linear_transformation(points=tvecs_similarity_correction, measurements=gt_tvecs)
    print('Ok.')
    print('Transform')
    print(Transform)
    
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
    
    traj_length = gt_trajectory_length[-1]
    ATE_percentage = relative_position_error(ate_mean_rigid, traj_length)
    print('Percentage Translation Error / Trajectory Length:', ATE_percentage, '/', traj_length)
    
    drawer.clear(window_name=WindowName.PosesComplete)
    drawer.draw(window_name=WindowName.PosesComplete, tvecs=tvec_normalized, color='#ff0000')
    drawer.draw(window_name=WindowName.PosesComplete, tvecs=gt_tvec_normalized, color='#0000ff')
    drawer.draw(window_name=WindowName.PosesComplete, tvecs=tvecs_similarity_correction_normalized, color='#00ffff')
    drawer.draw(window_name=WindowName.PosesComplete, tvecs=tvecs_homogeneous_correction_normalized, color='#00ff00')
    
    drawer.clear(window_name=WindowName.PosesEvaluation)
    drawer.draw(window_name=WindowName.PosesEvaluation, tvecs=gt_tvec_normalized, color='#0000ff')
    drawer.draw(window_name=WindowName.PosesEvaluation, tvecs=tvecs_homogeneous_correction_normalized, color='#00ff00')
    drawer.draw(window_name=WindowName.PosesEvaluation, tvecs=tvecs_similarity_correction_normalized, color='#00ffff')

    drawer.clear(window_name=WindowName.PosesComplete3d)
    drawer.draw(window_name=WindowName.PosesComplete3d, tvecs=tvec_normalized, color='#ff0000')
    drawer.draw(window_name=WindowName.PosesComplete3d, tvecs=gt_tvec_normalized, color='#0000ff')
    drawer.draw(window_name=WindowName.PosesComplete3d, tvecs=tvecs_similarity_correction_normalized, color='#00ffff')
    drawer.draw(window_name=WindowName.PosesComplete3d, tvecs=tvecs_homogeneous_correction_normalized, color='#00ff00')
    
    
    
    drawer.update()
    
    input('Press enter to plot on browser.')
       
        
    
    fig = BrowserDrawer()
    
    # draw 3D map pose
    fig.draw_points(tvecs, color=np.array([[255], [0], [0]]), name='Estimation')
    fig.draw_points(gt_tvecs, color=np.array([[0], [0], [255]]), name='Ground Truth')
    fig.draw_points(tvecs_similarity_correction, color=np.array([[0], [255], [255]]), name='Similarity')
    fig.draw_points(tvecs_homogeneous_correction, color=np.array([[0], [255], [0]]), name='Rigid')
    
    fig.show('test_trajectory_alignment')
    



if __name__ == "__main__":
    main()
    