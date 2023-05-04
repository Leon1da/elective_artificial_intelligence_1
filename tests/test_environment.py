
import sys
sys.path.append('/home/leonardo/elective_project/')
# sys.path.append('/home/leonardo/elective_project/simulation')


from simulation.environment import *
from drawer import *

cube_shape = generate_cuboid()
cube_shape = scale(cube_shape, 10)
cube_shape = move(cube_shape, np.array([20, 20, 20]))

torus_shape = generate_torus()
torus_shape = scale(torus_shape, 10)
torus_shape = move(torus_shape, np.array([5, 5, 5]))

sphere_shape = generate_sphere()
sphere_shape = scale(sphere_shape, 5)
sphere_shape = move(sphere_shape, np.array([10, 10, 10]))

shape = sphere_shape
num_sample = 1000

# samples = sampling_from_shape(shape=shape, num_points=num_sample, random_sampling=False, noise=False)

# samples = sampling_from_shape(shape=shape, num_points=num_sample, random_sampling=False, noise=True)

# samples = sampling_from_shape(shape=shape, num_points=num_sample, random_sampling=True, noise=False)

samples = sampling_from_shape(shape=shape, num_points=num_sample, random_sampling=True, noise=True)


fig_3d = BrowserDrawer()
fig_3d.draw_points(samples.T)
fig_3d.show()
