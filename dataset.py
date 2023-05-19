import os
import numpy as np
import pycolmap
#from hloc.utils.read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec
import imageio.v3 as iio
from utils.dataset_utils import *

class DatasetType:
    BIRD = "bird_data"
    ICRA = "icra_data"

class SfMDataset:
  def __init__(self, workdir, dataset_name=DatasetType.BIRD):
        
      self.dataset_path = workdir + "/data/" + dataset_name
      self.dataset_type = dataset_name
        
      if dataset_name == DatasetType.BIRD:
        # Folders:
        # - calib (.txt)
        # - images (.ppm)
        # - silhouettes (.pgm)

        fns = os.listdir(self.dataset_path + "/images")
        self.images_fn = sorted(list([self.dataset_path + "/images/" + fn for fn in fns])) 
        
        fns = os.listdir(self.dataset_path + "/silhouettes")
        self.silhouettes_fn = sorted(list([self.dataset_path + "/silhouettes/" + fn for fn in fns])) 
        
        fns = os.listdir(self.dataset_path + "/calib")
        self.calibs_fn = sorted(list([self.dataset_path + "/calib/" + fn for fn in fns])) 
      
      elif dataset_name == DatasetType.ICRA:
            # Folders:
            # - livingroom1-traj (.txt)
            # - livingroom1-color (.jpg)
            # - livingroom1-depth-clean (.png)
            fns = os.listdir(self.dataset_path + "/livingroom1-color")
            self.images_fn = sorted(list([self.dataset_path + "/livingroom1-color/" + fn for fn in fns])) 
            
            fns = os.listdir(self.dataset_path + "/livingroom1-depth-clean")
            self.depths_fn = sorted(list([self.dataset_path + "/livingroom1-depth-clean/" + fn for fn in fns])) 
            
            self.poses_fn = self.dataset_path + "/livingroom1-traj.txt"
            
            self.point_cloud_fn = self.dataset_path + "/livingroom.ply"

            # taken from the paper:
            # A Benchmark for {RGB-D} Visual Odometry, {3D} Reconstruction and {SLAM}
            # A. Handa and T. Whelan and J.B. McDonald and A.J. Davison

            self.calibration_matrix = np.array([[481.20, 0, 319.50], [0, -480.0, 239.50], [0, 0, 1]])
            # camera params
            self.fx, self.fy, self.px, self.py = 481.20, -480.0, 319.50, 239.50
            
            self.gt_cameras = {}
            self.gt_images = {}
            self.gt_images_depth = {}
            
      self.n = len(fns)

  def load_poses(self):
    if self.dataset_type == DatasetType.BIRD:
      # Read calibration data (camera poses)
      self.camera_poses = []
      for num, calib_fn in enumerate(self.calibs_fn):
        P = np.zeros((3, 4))
        with open(self.calib_fn) as f:
          lines = f.readlines()
          for i, line in enumerate(lines[1:]):
            line = line.split()
            P[i, 0], P[i, 1], P[i, 2], P[i, 3] = float(line[0]), float(line[1]), float(line[2]), float(line[3])
            self.camera_poses.append(P)
    
    elif self.dataset_type == DatasetType.ICRA:
      # Read calibration data (camera poses)
      self.camera_poses = np.zeros((self.n, 4, 4))

      with open(self.poses_fn) as f:
        lines = f.readlines()
        num_poses = int(len(lines) / 5)
        print("num_poses", num_poses)
        for i in range(num_poses):
          H = np.zeros((4, 4))
          # lines[i*5] # pose number (drop)
          row0 = lines[i*5 + 1].split()
          H[0, 0], H[0, 1], H[0, 2], H[0, 3] = float(row0[0]),float(row0[1]),float(row0[2]),float(row0[3])
          row1 = lines[i*5 + 2].split()
          H[1, 0], H[1, 1], H[1, 2], H[1, 3] = float(row1[0]),float(row1[1]),float(row1[2]),float(row1[3])
          row2 = lines[i*5 + 3].split()
          H[2, 0], H[2, 1], H[2, 2], H[2, 3] = float(row2[0]),float(row2[1]),float(row2[2]),float(row2[3])
          row3 = lines[i*5 + 4].split()
          H[3, 0], H[3, 1], H[3, 2], H[3, 3] = float(row3[0]),float(row3[1]),float(row3[2]),float(row3[3])
          
          self.camera_poses[i, :, :] = H
          
          
  # TODO sistemare tutto il Dataset loader
  def load_image(self, fn):
    image_id = path_to_id(fn)
    image_filename = self.images_fn[image_id]
    
    image = pycolmap.Image()
    
    # the reconstruction image ids start from 1
    # the dataset image ids start from 0
    image.image_id = image_id + 1 # in order to anchor the dataset image to the reconstruction images
    image.name = image_filename
    
    pose = self.camera_poses[image_id]
    image.qvec = rotmat2qvec(pose[:3, :3])
    image.tvec = pose[:3, 3]
    
    self.gt_images[image_id + 1] = image
    
    return image_filename 
    
    
  def load_image_data(self, fns):
    
    ids = np.array([path_to_id(fn) for fn in fns])
    for index, id in enumerate(ids):
      image = pycolmap.Image()
      image.image_id = index + 1
      image.name = fns[index]
      
      pose = self.camera_poses[id]
      image.qvec = rotmat2qvec(pose[:3, :3])
      image.tvec = pose[:3, 3]
      
      self.gt_images[index + 1] = image
      
  def load_depth_data(self, fn):
    
    image_id = path_to_id(fn)
    image_filename = self.depths_fn[image_id]
    
    depth_image = iio.imread(image_filename)
      
    # convert rgbd image to points in world frames
    height, width = depth_image.shape
    # compute indices:
    jj = np.tile(range(width), height)
    ii = np.repeat(range(height), width)
    # Compute constants:
    xx = (jj - self.px) / self.fx
    yy = (ii - self.py) / self.fy
    # transform depth image to vector of z:
    z = depth_image.reshape(height * width)
    # compute point cloud
    # 0.001 pixel to meter conversion (one pixel is one millimeter)
    length = height * width
    # points in camera frame
    scale = 0.001
    # scale = 1
    pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3)) * scale
    pcd = np.hstack((pcd, np.ones((length, 1))))
    
    pcd_valid = np.array([1 if depth else 0 for depth in z ]).reshape((height, width))
    
    
    # points in world frame
    homogeneous = self.camera_poses[image_id]
    pcd_world = np.array([homogeneous @ hom_vect for hom_vect in pcd])
    pcd_world = pcd_world[:, :3].reshape((height, width, 3))
    
    points3d_map = np.zeros((height, width), dtype=pycolmap.Point3D)
    for h in np.arange(height):
      for w in np.arange(width):
        point = pycolmap.Point3D()
        point.xyz = pcd_world[h, w]
        point.error = pcd_valid[h, w]
        points3d_map[h, w] = point
    
    self.gt_images_depth[image_id] = points3d_map
    # self.gt_images_depth[image_id + 1] = points3d_map
    return image_filename
    
  # Load all the depth data is unpracticable
  # The data is loaded only when required by the systems
   
  # def load_depth_data(self, fns):
  #   fns = sorted(fns)
  #   # parse input 
  #   fns = np.array([fn.split('/')[-1] for fn in fns]) # remove directory (/)
  #   fns = np.array([fn.replace('.jpg', '.png') for fn in fns]) # replace .jpg with .png
    
  #   for index, depth_fn in enumerate(fns):
  #     fn = self.dataset_path + "/livingroom1-depth-clean/" + depth_fn
  #     print(fn)
      
  #     depth_image = iio.imread(fn)
        
  #     # convert rgbd image to points in world frames
  #     height, width = depth_image.shape
  #     # compute indices:
  #     jj = np.tile(range(width), height)
  #     ii = np.repeat(range(height), width)
  #     # Compute constants:
  #     xx = (jj - self.px) / self.fx
  #     yy = (ii - self.py) / self.fy
  #     # transform depth image to vector of z:
  #     z = depth_image.reshape(height * width)
  #     # compute point cloud
  #     # 0.001 pixel to meter conversion (one pixel is one millimeter)
  #     length = height * width
  #     # points in camera frame
  #     scale = 0.001
  #     scale = 1
  #     pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3)) * scale
  #     pcd = np.hstack((pcd, np.ones((length, 1))))
      
  #     # points in world frame
  #     homogeneous = self.camera_poses[index]
  #     pcd_world = np.array([homogeneous @ hom_vect for hom_vect in pcd])[:, :3]
      
  #     self.gt_points3D[index + 1] = np.reshape(pcd_world, (height, width, 3))
      

    
