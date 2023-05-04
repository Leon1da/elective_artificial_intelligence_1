import numpy as np


def quaternion_to_rotation(quaternion):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from quaternion
    q0, q1, q2, q3 = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def quaternion_translation_to_rototranslation(quaternion, translation):
    rototranslation = np.zeros((3, 4))
    rototranslation[:3, :3] = quaternion_to_rotation(quaternion)
    rototranslation[:3, 3] = translation
    return rototranslation
    
def rotation_translation_to_rototranslation(rotation, translation):
    rototranslation = np.zeros((3, 4))
    rototranslation[:3, :3] = rotation
    rototranslation[:3, 3] = translation
    return rototranslation  

def rotation_translation_to_homogeneous(rotation, translation):
    homogeneous = np.eye(4)
    homogeneous[:3, :3] = rotation
    homogeneous[:3, 3] = translation
    return homogeneous

def homogeneous_to_rotation_translation(homogeneous):
    rotation = homogeneous[:3, :3]
    translation = homogeneous[:3, 3]
    return rotation, translation

def rototranslation_to_homogeneous(rototranslation):
    homogeneous = np.eye(4)
    homogeneous[:3, :3] = rototranslation[:3, :3]
    homogeneous[:3, 3] = rototranslation[:3, 3]
    return homogeneous

def homogeneous_to_rototranslation(homogeneous):
    rotation = homogeneous[:3, :3]
    translation = homogeneous[:3, 3]
    np.hstack([rotation, translation])
    return rotation, translation

def rotation_to_direction_vectors(rotation):
    dx, dy, dz = rotation
    return dx, dy, dz

def Rx(angle):
    R = np.zeros((3, 3))
    R[0, 0] = 1
    R[1, 1] = np.cos(angle)
    R[1, 2] = -np.sin(angle)
    R[2, 1] = np.sin(angle)
    R[2, 2] = np.cos(angle)
    return R

def Ry(angle):
    R = np.zeros((3, 3))
    R[1, 1] = 1
    R[0, 0] = np.cos(angle)
    R[0, 2] = np.sin(angle)
    R[2, 0] = -np.sin(angle)
    R[2, 2] = np.cos(angle)
    return R

def Rz(angle):
    R = np.zeros((3, 3))
    R[0, 0] =  np.cos(angle)
    R[0, 1] = -np.sin(angle)
    R[1, 0] = np.sin(angle)
    R[1, 1] = np.cos(angle)
    R[2, 2] = 1
    return R

# vector to skew symmetric matrix
def skew(v):
    S = np.zeros((3, 3))
    S[0, 1] = -v[2]
    S[1, 0] = v[2]
    
    S[0, 2] = v[1]
    S[2, 0] = -v[1]
    
    S[1, 2] = -v[0]
    S[2, 1] = v[0]
    
    return S    

# vector to similarity
def v2s(v):
    S = v2t(v[:6])
    S[3, 3] = np.exp(v[6])
    return S
  
# vector to transformation  
def v2t(v):
    T = np.eye(4)
    T[:3, :3] = Rx(v[3]) @ Ry(v[4]) @ Rz(v[5]) 
    T[:3, 3] = v[:3, :].squeeze()
    return T


def transform(T, v):
    R = T[:3, :3]
    t = T[:3, 3]
    return R @ v + t

def similarity_transform(S, v):
    s = S[3, 3]
    R = S[:3, :3]
    t = S[:3, 3]
    return s * (R @ v + t)

    
