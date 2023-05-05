
import sys
sys.path.append('/home/leonardo/elective_artificial_intelligence_1/')


from simulation.environment import *
from drawer import *


fig_3d = BrowserDrawer()

num_sample = 1000

cube_shape = generate_cuboid()
cube_shape = sample(shape=cube_shape, num_points=1000, random_sampling=True)
cube_shape = add_white_gaussian_noise(cube_shape, mu=0, sigma=0.1)
# cube_shape = move(cube_shape, np.array([20, 20, 20]))
cube_shape = scale(cube_shape, 10)
cube_shape = cube_shape.T

torus_shape = generate_torus()
torus_shape = sample(shape=torus_shape, num_points=1000, random_sampling=True)
torus_shape = add_white_gaussian_noise(torus_shape, mu=0, sigma=0.01)
torus_shape = move(torus_shape, np.array([2, 2, 2]))
torus_shape = scale(torus_shape, 10)
torus_shape = torus_shape.T

sphere_shape = generate_sphere()
sphere_shape = sample(shape=sphere_shape, num_points=1000, random_sampling=True)
sphere_shape = add_white_gaussian_noise(sphere_shape, mu=0, sigma=0.1)
sphere_shape = move(sphere_shape, np.array([5, 5, 5]))
sphere_shape = scale(sphere_shape, 5)
sphere_shape = sphere_shape.T


color = np.array([255, 0, 0]).reshape(3, 1)
fig_3d.draw_points(cube_shape, color=color, name='cube')

color = np.array([0, 255, 0]).reshape(3, 1)
fig_3d.draw_points(torus_shape, color=color, name='torus')

color = np.array([0, 0, 255]).reshape(3, 1)
fig_3d.draw_points(sphere_shape, color=color, name='sphere')

fig_3d.show()
