import numpy as np

def get_coordinates(points):
    return np.array([[int(point.xy[0]), int(point.xy[1])] for point in points]) 

def get_point3D_ids(points):
    return np.array([point.point3D_id for point in points])

