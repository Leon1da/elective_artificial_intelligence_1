# import tqdm, tqdm.notebook
# tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
import pycolmap

from tqdm import tqdm
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

images = Path('/mnt/d/data/icra_data/livingroom1-color/')

outputs = Path('/home/leonardo/elective_project/reconstruction_output')

# TODO set the model folder (reconstruction_folder)
reconstruction_folder = 'output_002'
outputs = outputs / reconstruction_folder
 

# !rm -rf $outputs
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'



# # Default configuration
# feature_conf = extract_features.confs['superpoint_aachen']
# matcher_conf = match_features.confs['superglue']

# feature_conf = extract_features.confs['sift']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue-fast']


references = [p.relative_to(images).as_posix() for p in images.iterdir()]
print("Found", len(references), "images.")

# plot_images([read_image(images / r) for r in references[:4]], dpi=50)
num_image = 200
references = references[:num_image]
print(len(references), "images will be used for the reconstruction.")
# print(references)


extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

# define CAMERA_MODEL_DEFINITIONS(model_id_value, model_name_value, num_params_value)  
# camera_mode = (1, "PINHOLE", 4)
# TODO modify 
# - image_options and mapper options
# to tackle camera models and refine behaviours
model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.PINHOLE, image_list=references)
# model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
fig.show()

visualization.visualize_sfm_2d(model, images, color_by='visibility', n=2)