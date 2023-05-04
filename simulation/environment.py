import numpy as np

class DataDistribution:
    CUBOID = "CUBOID"
    SPHERE = "SPHERE"
    TORUS = "TORUS"

# 3D Environment generation
class Environment:
  
  def __init__(self, 
               boundaries=[[0, 100], [0, 100], [0, 100]], 
               num_points=100, 
               num_cameras=10, 
               points_distribution=DataDistribution.CUBOID, 
               ) -> None:
    
    self.min_x, self.max_x = boundaries[0]
    self.min_y, self.max_y = boundaries[1]
    self.min_z, self.max_z = boundaries[2]
    
    self.num_points = num_points
    self.num_cameras = num_cameras
    
    cx = (self.min_x + self.max_x)/2
    cy = (self.min_y + self.max_y)/2
    cz = (self.min_z + self.max_z)/2

    if points_distribution == DataDistribution.CUBOID:
      print(DataDistribution.CUBOID)
      shape = generate_cuboid()
    elif points_distribution == DataDistribution.SPHERE:
      print(DataDistribution.SPHERE)
      shape = generate_sphere()
    elif points_distribution == DataDistribution.TORUS:
      print(DataDistribution.TORUS)
      shape = generate_torus()
    
    shape = scale(shape, 10)
    # shape = move(shape, np.array([cx, cy, cz]))
    
    self.points_position = sampling_from_shape(shape=shape, num_points=num_points, random_sampling=True, noise=True)

        
  

def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

def move(vec, _position):
  return vec + _position.reshape(-1, 1)

def scale(vec, _scale):
  return vec * _scale

def rotate(vec, _rotation):
  return _rotation @ vec

def transform(vec, _scale, _rotation, _translation):
    vec = rotate(vec, _rotation)
    vec = scale(vec, _scale)
    vec = move(vec, _translation)
    return vec

def sample(shape, num_points, random_sampling=False):
  num_shape_points = shape.shape[1]
  indices = np.arange(num_shape_points)
  if random_sampling:
    idx = np.random.choice(indices, num_points)
  else:
    # TODO since the shape's mesh is reshaped after the creation 
    # in order to have equally distribuited point the step definition should be slightly  different
    step = int(num_shape_points/num_points)
    idx = np.arange(0, num_shape_points, step)
    
  samples = shape[:, idx]
  return samples


def add_white_gaussian_noise(samples, mu=0, sigma=1):
  white_gaussian_noise = np.random.normal(mu, sigma, samples.shape)
  samples = samples + white_gaussian_noise
  return samples
  
def sampling_from_shape(shape, num_points, random_sampling=False, noise=False):
  
  num_shape_points = shape.shape[1]
  indices = np.arange(num_shape_points)
  if random_sampling:
    idx = np.random.choice(indices, num_points)
  else:
    # TODO since the shape's mesh is reshaped after the creation 
    # in order to have equally distribuited point the step definition should be slightly  different
    step = int(num_shape_points/num_points)
    idx = np.arange(0, num_shape_points, step)
    
  samples = shape[:, idx]
  if noise:
    mu, sigma = 0, 1 # mean and standard deviation
    white_gaussian_noise = np.random.normal(mu, sigma, samples.shape)
    samples = samples + white_gaussian_noise

  return samples

# TORUS equation
# x = (c + a*cosθ)*cosϕ
# y = (c + a*cosθ)*sinϕ
# z = a*sinθ
def generate_torus(torus_center = [0, 0, 0], torus_radius=0.5, torus_internal_radius=0.1):
  
  n = 100
  theta = np.linspace(0, 2.*np.pi, n)
  phi = np.linspace(0, 2.*np.pi, n)
  theta, phi = np.meshgrid(theta, phi)
  
  x = torus_center[0] + (torus_radius + torus_internal_radius*np.cos(theta)) * np.cos(phi)
  y = torus_center[1] + (torus_radius + torus_internal_radius*np.cos(theta)) * np.sin(phi)
  z = torus_center[2] +  torus_internal_radius * np.sin(theta)
  
  x = x.reshape(1, -1)
  y = y.reshape(1, -1)
  z = z.reshape(1, -1)
  
  torus = np.vstack([x, y, z])

  return torus

# SPHERE equation
# x = cosθ*sinϕ
# y = sinθ*sinϕ
# z = cosϕ
def generate_sphere(sphere_center = [0, 0, 0], sphere_radius=0.5):
  n = 100
  theta = np.linspace(0, 2.*np.pi, n)
  phi = np.linspace(0, 2.*np.pi, n)
  theta, phi = np.meshgrid(theta, phi)
  x = sphere_center[0] + sphere_radius*np.cos(theta)*np.sin(phi)
  y = sphere_center[1] + sphere_radius*np.sin(theta)*np.sin(phi)
  z = sphere_center[2] + sphere_radius*np.cos(phi)

  x = x.reshape(1, -1)
  y = y.reshape(1, -1)
  z = z.reshape(1, -1)
  
  sphere = np.vstack([x, y, z])
  return sphere

def generate_cuboid(cube_center = [0, 0, 0], width=1, heigth=1, depth=1):
  n = 10
  
  x = np.linspace(-width/2, width/2, n)
  y = np.linspace(-depth/2, depth/2, n)
  z = np.linspace(-heigth/2, heigth/2, n)
  
  # generate each possible permutation
  cube = np.array([[cube_center[0] + i, cube_center[0] + j, cube_center[0] + k] for i in x for j in y for k in z]).T
  
  return cube

