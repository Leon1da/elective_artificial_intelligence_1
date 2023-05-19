import sys, os
sys.path.append(os.getcwd())

from simulation.environment import *
from scale_estimator import ScaleEstimatorModule

from evaluation import absolute_position_error, absolute_rotation_error
from utils.geometric_utils import *
from drawer import *


fig_3d = BrowserDrawer()

    
# generate a torus point cloud
shape = generate_torus()
# shape = generate_cuboid()
# shape = generate_sphere()
pc1 = sample(shape=shape, num_points=100, random_sampling=True)
pc1 = scale(pc1, 10) # it is scaled for a better visualizzation


# generate a measurements with noise for the torus point cloud
pc2 = pc1
pc2 = add_white_gaussian_noise(pc1, mu=0, sigma=5)       # add noise
pc2 = move(pc2, np.array([5, 5, 5]))                        # add translation
pc2 = scale(pc2, 2)                                         # add scale
pc2 = rotate(pc2, Rx(np.pi/4))                              # add rotation

pc1 = pc1.T
pc2 = pc2.T

print("points shape", pc1.shape)
print("measurements shape", pc2.shape)


estimator = ScaleEstimatorModule()
estimator.configure(iterations=2000, dumping=0.7, kernel_threshold=0.01)
Similarity, chi_evolution, num_inliers_evolution, similarity_evolution = estimator.recover_similarity_transformation(points=pc1, measurements=pc2)

print(Similarity)


pc_similarity_correction = np.array([similarity_transform(np.linalg.inv(Similarity), p) for p in pc2])

        
# Compute the Error
similarity_position_error = absolute_position_error(pc_similarity_correction, pc1)
print('Mean Square Error (Similarity):', similarity_position_error)

# similarity_rotation_error = absolute_rotation_error(Rx(np.pi/4), np.eye(3))
# print('Mean Square Error (Similarity):', similarity_position_error)

estimator.configure(iterations=1500, dumping=0.7, kernel_threshold=0.01)
Homogeneous, chi_evolution, num_inliers_evolution, homogeneous_evolution = estimator.recover_rigid_transformation(points=pc1, measurements=pc_similarity_correction)

print(Homogeneous)

pc_homogeneous_correction = np.array([similarity_transform(np.linalg.inv(Homogeneous), p) for p in pc_similarity_correction])

# Compute the Error
homogeneous_error = absolute_position_error(pc_homogeneous_correction, pc1)
print('Mean Square Error (Homogeneous):', homogeneous_error)
       

color = np.array([0, 0, 255]).reshape(3, 1)
fig_3d.draw_points(pc1, color=color, name='points')

color = np.array([255, 0, 0]).reshape(3, 1)
fig_3d.draw_points(pc2, color=color, name='measurements')

color = np.array([0, 255, 255]).reshape(3, 1)
fig_3d.draw_points(pc_similarity_correction, color=color, name='sim correction')

color = np.array([0, 255, 0]).reshape(3, 1)
fig_3d.draw_points(pc_homogeneous_correction, color=color, name='hom correction')

fig_3d.show('test_scale_estimation_module')
