# elective_artificial_intelligence_1

# test_simulator_generation.py
generate 3 point clouds:
- a torus 
- a sphere
- a cuboid
with different level of noise

# test_scale_estimator_module.py
1. generate a point clouds (blue)
2. generate a sets of measurements for the point cloud (red)
3. estimate the similarity that aligneate the two point clouds (cyan)
4. refine the obtained solution estimating an linear transformation between the points and the measurements corrected with the similarity above (green)

# test_sfm_module.py
given a set of images
1. extract features
2. compute matches
3. run a reconstruction

# test_sfm_visualization.py
1. visualize a reconstruction