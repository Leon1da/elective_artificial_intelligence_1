import numpy as np
from utils.geometric_utils import *

def absolute_position_error(tvec, gt_tvec):
    tvec_diff = tvec - gt_tvec
    tvec_norm = np.linalg.norm(tvec_diff)
    tvec_norm = np.mean(tvec_norm)
    return tvec_norm

def absolute_rotation_error(rotation, gt_rotation):
    
    rotation_angle = np.array([rotation_to_axis_angle(r)[1] for r in rotation])
    gt_rotation_angle = np.array([rotation_to_axis_angle(gt_r)[1] for gt_r in gt_rotation])
    
    rotation_diff = rotation_angle - gt_rotation_angle
    rotation_norm = np.linalg.norm(rotation_diff)
    rotation_norm = np.mean(rotation_diff)
    
    return rotation_norm