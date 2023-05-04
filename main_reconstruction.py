# 
# This file perform the reconstruction having the matches and the features already extracted.
# The base reconstruction folder should be passed as input
# 
from pathlib import Path
import argparse

import numpy as np
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.utils import viz_3d

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COLMAP binary and text models")
    parser.add_argument("--input_model", required=True, help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    outputs = Path(args.input_model)
    
    images = Path('/mnt/d/data/icra_data/')

    # !rm -rf $outputs
    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sfm'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    references = np.array(sorted([str(p.relative_to(images)) for p in (images / 'livingroom1-color/').iterdir()]))
    # references = np.array(sorted('livingroom1-color' / os.listdir(images / 'livingroom1-color')))
    num_references = len(references)
    print(num_references, "mapping images found.")
    print(references)
    print()

    # We dont load all the images since the performance drop down significantly
    num_images = 250 # sample N image from the set

    # # sequentially
    # idx = np.arange(num_images) # sample the first 'num_images' images

    # step
    step = 5
    idx = np.arange(0, num_images * step, step) # sample an image every 'step' images, a total of 'num_images' images will be sampled

    # and soon
    # idx = np.arange(0, num_images, 10) # sample an image every 5 images, a total of 'num_images' images will be sampled

    references = references[idx].tolist()
    num_references = len(references)
    print(num_references, "mapping images taken.")
    # print(references)
    
    
    model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)

    # fig = viz_3d.init_figure()
    # viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
    # fig.show()


    # visualization.visualize_sfm_2d(model, images, color_by='visibility', n=2)



if __name__ == "__main__":
    main()