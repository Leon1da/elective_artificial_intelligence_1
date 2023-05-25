# Elective in Artificial Intelligence: AI for Visual Perception in HCI & HRI

This repo contains the code for the project **Beacon-based Scale Estimation for Monocular Structure from Motion in Robotics Systems**



## Installation

```
pip install -r requirements.txt
```

Tested on Ubuntu 20.04 LTS + Python 3.8

#### Optional:
Install COLMAP (https://github.com/colmap/colmap) and hloc (https://github.com/cvg/Hierarchical-Localization) to start the mapping phase. *Note that some maps are already provided with the dataset.*

### Data
- Unzip dataset.zip inside data/icra_data/

## Run the complete pipelines

There are two ways to run the pipeline:


Incrementally adjust the scale using the available data coming from the beacons:
```
python tests/test_incremental_pipeline.py --input_model reconstruction_outputs/<recostruction_sequence>/sfm/
```


Compute the scale correction oneshot:
```
python tests/test_oneshot_pipeline.py --input_model reconstruction_outputs/<recostruction_sequence>/sfm/
```



## Test building block

### test_scale_estimator_module.py
1. generate a point clouds (blue)
2. generate a sets of measurements for the point cloud (red)
3. estimate the Similarity transformation that aligns the two point clouds (cyan)
4. refine the obtained solution estimating a Rigid transformation between the points and the measurements corrected with the Similarity above (green)

```
python tests/test_scale_estimator_module.py
```

### test_segmentation_module.py
Segment the given images.

```
python tests/test_segmentation_module.py
```

### test_sfm_module.py
given a set of images
1. extract features
2. compute matches
3. run a reconstruction


```
python tests/test_sfm_module.py
```

**Note**: 
- hloc required
- in order to run a custom reconstruction the file test_sfm_module.py should be properly cofigurated.

### test_sfm_visualization.py
1. visualize a reconstruction

```
python tests/test_sfm_visualization.py --input_model reconstruction_outputs/<recostruction_sequence>/sfm/
```

## Validation

### test_simulator_generation.py
generate 3 point clouds:
- a torus 
- a sphere
- a cuboid
with different level of noise

```
python tests/test_simulator_generation.py
```

**Note**: 
- in order to generate a torus, a sphere or a cuboid the file test_simulator_generation.py should be properly modified.

### test_trajectory_alignment.py
given a reconstruction
1. perform Iterative Closest Point optimization using the ground-truth poses and estimated poses

```
python tests/test_trajectory_alignment.py --input_model reconstruction_outputs/<recostruction_sequence>/sfm/
```
