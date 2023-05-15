import numpy as np
from utils.geometric_utils import *

def absolute_position_error(tvec, gt_tvec):
    tvec_diff = tvec - gt_tvec
    tvec_norm = np.linalg.norm(tvec_diff, axis=1)
    tvec_mean = np.mean(tvec_norm)
    tvec_std = np.std(tvec_norm)
    tvec_min = np.min(tvec_norm)
    
    return tvec_mean, tvec_std, tvec_min

def absolute_rotation_error(rotation, gt_rotation):
    
    # rotation_angle = np.array([rotation_to_axis_angle(r)[1] for r in rotation]).reshape(-1, 1)
    # gt_rotation_angle = np.array([rotation_to_axis_angle(gt_r)[1] for gt_r in gt_rotation]).reshape(-1, 1)
    
    # rotation_diff = rotation_angle - gt_rotation_angle
    # rotation_norm = np.linalg.norm(rotation_diff, axis=1)
    # rotation_norm = np.mean(rotation_norm)
    
    relative_rotation = np.array([gt_rot @ rot.T for gt_rot, rot in zip(gt_rotation, rotation)])
    relative_angle = np.array([rotation_to_axis_angle(r)[1] for r in relative_rotation]).reshape(-1, 1)
    rotation_norm = np.linalg.norm(relative_angle)
    rotation_mean = np.mean(rotation_norm)    
    rotation_std = np.std(rotation_norm)    
    rotation_min = np.min(rotation_norm)    
    
    return rotation_mean, rotation_std, rotation_min

def relative_position_error(absolute_position_error, trajectory_length):
    # absolute_position_error : trajectory_length = percentage_error : 100
    percentage_error = absolute_position_error * 100 / trajectory_length
    return percentage_error

def mean_relative_error(gt_tvec, est_tvec):
    pass