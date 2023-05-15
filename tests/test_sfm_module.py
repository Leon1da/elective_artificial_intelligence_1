import sys, os
sys.path.append(os.getcwd())

from pathlib import Path
from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive

import tqdm
import numpy as np

workdir = os.getcwd()
    
images = Path(workdir + '/data/icra_data/')
outputs = Path('reconstruction_output/')

# IMPORTANT SET THE NEW FOLDER
reconstruction_folder = 'output_1000_1900_step_3'
outputs = outputs / reconstruction_folder

print("Output folder", outputs)

# !rm -rf $outputs
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

feature_conf = extract_features.confs['superpoint_aachen']
# matcher_conf = match_features.confs['superglue-fast']
matcher_conf = match_features.confs['superglue']

references = np.array(sorted([str(p.relative_to(images)) for p in (images / 'livingroom1-color/').iterdir()]))
# references = np.array(sorted('livingroom1-color' / os.listdir(images / 'livingroom1-color')))
num_references = len(references)
print(num_references, "mapping images found.")
print(references)
print()

# We dont load all the images since the performance drop down significantly

num_images = 301 # sample N image from the set
start_index = 1000
# # sequentially
# idx = np.arange(start_index, start_index + num_images) # sample the first 'num_images' images

# step
step = 3
idx = np.arange(start_index, start_index + num_images * step, step) # sample an image every 'step' images, a total of 'num_images' images will be sampled

references = references[idx].tolist()
num_references = len(references)
print(num_references, "mapping images taken.")
print(references)
print("Start image", references[0])
print("End image", references[-1])


# Extract image features
extract_features.main(feature_conf, images, image_list=references, feature_path=features)

# Compute Image pairs for matching
pairs_from_exhaustive.main(sfm_pairs, image_list=references)

# Match features among images pairs
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)


# Configure reconstruction
camera_params = "481.2,480.0,319.5,239.5"
# camera_params = []
image_options= {  
    'camera_model' : 'PINHOLE',
    'camera_params' : camera_params
    
}

mapper_options = {
    'ba_refine_focal_length' : False,
    'ba_refine_principal_point' : False,
    'ba_refine_extra_params' : False
}

# Compute reconstruction
model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references, mapper_options = mapper_options, image_options = image_options)
# model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references, mapper_options = mapper_options, image_options = image_options, skip_geometric_verification = True)
