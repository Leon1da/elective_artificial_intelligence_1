import numpy as np

def remove_extension(filename):
    return filename.split('.')[0] # remove extension (.jpg)

def path_to_filename(path):
    return path.split('/')[-1] # remove directory (/)
    
def path_to_id(path):
    filename = path_to_filename(path)
    filename = remove_extension(filename)
    filename = int(filename)
    return filename


def compute_total_trajectory_displacement(gt_tvecs):
    # compute the total displacemnt (from first pose to last pose)
    num_tvecs = gt_tvecs.shape[0]
    gt_trajectory_length = [0]
    for i in np.arange(num_tvecs - 1):
        actual_displacement = gt_trajectory_length[-1]
        displacement = np.linalg.norm(gt_tvecs[i] - gt_tvecs[i + 1]) * 10
        total_displacement = actual_displacement + displacement
        gt_trajectory_length.append(total_displacement)
    return gt_trajectory_length
    