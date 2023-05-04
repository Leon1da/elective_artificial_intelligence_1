import sys
sys.path.append('/home/leonardo/elective_project/')
# sys.path.append('/home/leonardo/elective_project/simulation')

from least_squares import *
from simulation.environment import *
from utils.geometric_utils import *
from drawer import *

torus_shape = generate_torus()
pc1 = sample(shape=torus_shape, num_points=1000, random_sampling=True)
pc1 = scale(pc1, 10)


pc2 = pc1
pc2 = add_white_gaussian_noise(pc1, mu=0, sigma=0.01)
pc2 = move(pc2, np.array([5, 5, 5]))
pc2 = scale(pc2, 2)
pc2 = rotate(pc2, Rx(np.pi/4))

print(pc1.shape)
print(pc2.shape)

S_guess = np.eye(4)
Similarity, chi_stats, error_s = robust_sicp(X_guess=S_guess, points=pc1.T, measurements=pc2.T, n_iterations=150, damping=0.7, kernel_threshold=0.1)
# Similarity, chi_stats = sicp(X_guess=S_guess, points=pc1.T, measurements=pc2.T, n_iterations=75)
print(Similarity)
print(chi_stats)


_s, _r, _t = Similarity[3, 3], Similarity[:3, :3], Similarity[:3, 3]
pc4 = transform(pc1, _s, _r, _t)

X_guess = np.eye(4)
X_guess[:3, :3] = _r
X_guess[:3, 3] = _t
Affine, chi_stats, error_t = robust_micp(X_guess=X_guess, points=pc4.T, measurements=pc2.T, n_iterations=100, damping=0.7, kernel_threshold=0.1)
# Affine, chi_stats = micp(X_guess=X_guess, points=pc4.T, measurements=pc2.T, n_iterations=50)
print(Affine)
print(chi_stats)

_s, _r, _t = Affine[3, 3], Affine[:3, :3], Affine[:3, 3]

pc5 = transform(pc4, _s, _r, _t)


fig_3d = Drawer()
fig_3d.plot_error(error=error_s)

fig_3d = Drawer()
fig_3d.plot_error(error=error_t)


fig_3d = BrowserDrawer()
fig_3d.draw_points(pc1.T, color=np.array([[255], [0], [0]]), name='pc1')
fig_3d.draw_points(pc2.T, color=np.array([[0], [0], [255]]), name='pc2')

# fig.draw_points(pc3.T, color=[0, 0, 0], name='pc3')
fig_3d.draw_points(pc4.T, color=np.array([[0], [0], [0]]), point_size=1, name='pc4')
fig_3d.draw_points(pc5.T, color=np.array([[0], [0], [0]]), point_size=1, name='pc5')
fig_3d.show()
