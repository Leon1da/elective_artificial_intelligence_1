import sys
sys.path.append('/home/leonardo/elective_artificial_intelligence_1/')

import numpy as np
from pathlib import Path
import pycolmap

workdir = '/mnt/d'

images = Path(workdir + '/data/icra_data/livingroom1-color/')
outputs = Path('reconstruction_output/')

# IMPORTANT SET THE NEW FOLDER
reconstruction_folder = 'output_simple'
outputs = outputs / reconstruction_folder

print("Output folder", outputs)

references = np.array(sorted([str(p.relative_to(images)) for p in images.iterdir()]))
# references = np.array(sorted('livingroom1-color' / os.listdir(images / 'livingroom1-color')))
num_references = len(references)
print(num_references, "mapping images found.")
print(references)
print()

# step
step = 5
start_index = 0
idx = np.arange(start_index, num_references, step) # sample an image every 'step' images, a total of 'num_images' images will be sampled


image_list = references[idx]
print(image_list)

outputs.mkdir(exist_ok=True)
mvs_path = outputs / "mvs"
database_path = outputs / "database.db"

print('Feature Extraction..')
pycolmap.extract_features(database_path, images, image_list, sift_options={"max_num_features": 512})

print('Feature Matching..')
pycolmap.match_exhaustive(database_path)

print('Incremental Mapping..')
maps = pycolmap.incremental_mapping(database_path, images, outputs)
maps[0].write(outputs)