# elective_artificial_intelligence_1

### Test of each building block

# test_scale_estimator_module.py
1. generate a point clouds (blue)
2. generate a sets of measurements for the point cloud (red)
3. estimate the similarity that aligneate the two point clouds (cyan)
4. refine the obtained solution estimating an linear transformation between the points and the measurements corrected with the similarity above (green)

# test_segmentation_module.py
Segment the given images.

# test_sfm_module.py
given a set of images
1. extract features
2. compute matches
3. run a reconstruction

# test_sfm_visualization.py
1. visualize a reconstruction

### Validation

# test_simulator_generation.py
generate 3 point clouds:
- a torus 
- a sphere
- a cuboid
with different level of noise

# test_trajectory_alignment.py
given a reconstruction
1. perform Iterative Closest Point optimization using the ground-truth poses and estimated poses

### Test the complete pipelines
# test_incremental_pipelines.py
Incrementally adjust the scale using the available data caming from the beacons

# test_oneshot_pipelines.py
Compute the scale drift correction once.