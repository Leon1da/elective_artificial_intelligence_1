import numpy as np
import scipy as scipy 


def reprojection_error(x, y):
    
    eu = x[0, :] - y[0, :]
    ev = x[1, :] - y[1, :]
    
    e = np.vstack([eu, ev])
    e = np.linalg.norm(e, axis=0)
    return e
  
# x1.shape = (3, n) (homogeneous [u, v, 1])
# x2.shape = (3, n) (homogeneous [u, v, 1])
def direct_linear_transformation(x1, x2):
    n = x1.shape[1]
    A = np.zeros((3*n, 9))
    for i in np.arange(n):
        x1_ = x1[:, i]
        x2_ = x2[:, i]
        
        Ai = np.zeros((3, 9))
        Ai[0, 3:6] = -x2_[2]*x1_
        Ai[0, 6:] =  x2_[1]*x1_
        
        Ai[1, :3] = x2_[2]*x1_
        Ai[1, 6:] = -x2_[0]*x1_
          
        Ai[2, :3] = -x2_[1]*x1_
        Ai[2, 3:6] = -x2_[0]*x1_
        A[3*i:3*i+3, :] = Ai

        
    U, D, Vw = np.linalg.svd(A)
    h = Vw[-1,:]
    H = h.reshape(3, 3)  
    return H 
    
def compute_normalizing_transformation(x):
    n = x.shape[1] 
    xm = np.mean(x, axis=1)
    s = np.sqrt(2) / (np.sum([np.linalg.norm(x_ - xm) for x_ in x.T]) / n)
    tx = -s*xm[0]
    ty = -s*xm[1]
    
    T = np.eye(3)
    T[0, 0] = s
    T[1, 1] = s
    T[0, 2] = tx
    T[1, 2] = ty
    return T
    
def normalized_direct_linear_transformation(x1, x2):
    T1 = compute_normalizing_transformation(x1)
    T2 = compute_normalizing_transformation(x2)
    
    print(x1.shape)
    x1_normalized = np.array([T1 @ x1_ for x1_ in x1.T]).T
    x2_normalized = np.array([T2 @ x2_ for x2_ in x2.T]).T
    H_normalized = direct_linear_transformation(x1_normalized, x2_normalized)
    H = np.linalg.inv(T2) @ H_normalized @ T1
    return H

# given a measured point correspondences x1 <-> x2 and a fundamental matrix F
# compute the corrected correspondences x1_hat <-> x1_hat 
# that minimize the geometric error subject to the epipolar constraint.  
def optimal_triangulation_method(x1, x2, F):
    # (i) Define the transformation matrices
    u1, v1 = x1
    u2, v2 = x2
    T1, T2 = np.eye(3), np.eye(3)
    T1[0, 2], T1[1, 2] = -u1, -v1
    T2[0, 2], T2[1, 2] = -u2, -v2
    
    # (ii) Replace the F by the F in traslated coordinates
    F = np.linalg.inv(T2).T @ F @ np.linalg.inv(T1)
    
    # (iii) Compute the right and left epipoles e1 and e2.
    # Normalize the epipoles such that e11**2 + e12**2 = 1 and e21**2 + e22**2 = 1
    U, D, Vw = np.linalg.svd(F)
    e1 = Vw[-1, :].reshape(-1, 1)
    e2 = U[:, -1].reshape(-1, 1)
    
    # # check consistency
    # print(e2.T @ F)     # e2.T * F = 0
    # print(F @ e1)       # F * e1 = 0
    
    e1 = e1 * (1 / (e1[0]**2 + e1[1]**2))
    e2 = e2 * (1 / (e2[0]**2 + e2[1]**2))
    # print("e1", e1) 
    # print("e2", e2)
    # print()
    
    # print(e1[0]**2 + e1[1]**2)
    # print(e2[0]**2 + e2[1]**2)
    
    # (iv) Form matrices R1 and R2 and observe that they are rotation matrices
    R1 = np.eye(3)
    R1[0, 0], R1[0, 1], R1[1, 0], R1[1, 1] = e1[0], e1[1], -e1[1], e1[0]
      
    R2 = np.eye(3)
    R2[0, 0], R2[0, 1], R2[1, 0], R2[1, 1] = e2[0], e2[1], -e2[1], e2[0]
    
    # print(np.linalg.det(R1))
    # print(R1 @ e1)
    # print(np.linalg.det(R2))
    # print(R2 @ e2)
    # print()
    
    # (v)
    F = R2 @ F @ R1.T
    # print(F)
    # print()
    
    # (vi)
    f1, f2, a, b, c, d = float(e1[2]), float(e2[2]), F[1, 1], F[1, 2], F[2, 1], F[2, 2]
    
    # print(f1, f2, a, b, c, d)
    # print()
    
    # (vii) 
    # g = t*((a*t + b)**2 + f2**2*(c*t + d)**2)**2 - (a*d - b*c)(1+f1**2*t**2)**2*(a*t+b)*(c*t+d)
    
    p6 = a*b*c**2*f1**4 -a**2*c*d*f1**4 #   6
    p5 = c**4*f2**4 + b**2*c**2*f1**4-a**2*d**2*f1**4 + 2*a**2*c**2*f2**2 + a**4 #   5 
    p4 = 4*c**3*d*f2**4 + b**2*c*d*f1**4 -a*b*d**2*f1**4 + 4*a*b*c**2*f2**2 + 2*a*b*c**2*f1**2 + 4*a**2*c*d*f2**2 -2*a**2*c*d*f1**2 + 4*a**3*b #   4 
    p3 = 6*c**2*d**2*f2**4 + 2*b**2*c**2*f2**2 + 2*b**2*c**2*f1**2 + 8*a*b*c*d*f2**2 + 2*a**2*d**2*f2**2 -2*a**2*d**2*f1**2 + 6*a**2*b**2 #   3 
    p2 = 4*c*d**3*f2**4 + 4*b**2*c*d*f2**2 + 2*b**2*c*d*f1**2 +4*a*b*d**2*f2**2 -2*a*b*d**2*f1**2 + a*b*c**2 +4*a*b**3 -a**2*c*d #   2 
    p1 = d**4*f2**4 + 2*b**2*d**2*f2**2 + b**2*c**2 + b**4 -a**2*d**2 #   1 
    p0 = b**2*c*d + a*b*d**2 # 0
    
    p = [p6, p5, p4, p3, p2, p1, p0]
    p = np.array([p6, p5, p4, p3, p2, p1, p0])
    rs = np.roots(p)
    # print(rs[0], rs[1], rs[2], rs[3], rs[4], rs[5])
    
    # (viii)
    cost_values = []
    for root in rs:
        t = root.real
        val = ((t**2) / (1 + f1**2*t**2)) + (((c*t + d)**2) / ((a*t + b)**2 + f2**2*(c*t + d)**2))
        cost_values.append(val)
      
    cost_values = np.array(cost_values)
    # print(cost_values) 
    t_min = rs[np.argmin(cost_values)]
    # print(t_min)
    
    asymptotic_value = 1/f1**2 + c**2/(a**2 + f2**2*c**2) 
    # print(asymptotic_value)
    
    # (ix)
    l1 = [t_min*f1, 1, -t_min]
    l2 = [-f2*(c*t_min + d), a*t_min + b, c*t_min + d]
    # print(l1, l2)
    
    # closes point on the line to the origin
    x1_hat = np.array([-l1[0]*l1[2], -l1[1]*l1[2], l1[0]**2+l1[1]**2])
    x2_hat = np.array([-l2[0]*l2[2], -l2[1]*l2[2], l2[0]**2+l2[1]**2])
    
    x1_hat = np.linalg.inv(T1) @ R1.T @ x1_hat
    x2_hat = np.linalg.inv(T2) @ R2.T @ x2_hat
    
    x1_hat = np.real(x1_hat)
    x2_hat = np.real(x2_hat)
    x1_hat = x1_hat/x1_hat[2]
    x2_hat = x2_hat/x2_hat[2]
    return x1_hat, x2_hat

def vector_to_skew_symmetric_matrix(v):
    a1, a2, a3 = v
    return np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

# compute the fundamental matrix 
# given 
# the camera calibration of each camera
# the roto-translation from Camera 1 to Camera 2 (relative transformation)
# def compute_fundamental_matrix(K1, K2, R, t):
#     F = np.linalg.inv(K2).T @ R @ K1.T @ vector_to_skew_symmetric_matrix(K1 @ R.T @ t)
#     return F

# def compute_fundamental_matrix(K, R, t):
#     iK = np.linalg.inv(K)
#     F = iK.T @ R @ K.T @ vector_to_skew_symmetric_matrix(K @ R.T @ t)
#     return F
def compute_fundamental_matrix(K, E):
    iK = np.linalg.inv(K)
    F = iK.T @ E @ iK
    return F

def compute_essential_matrix(R, t):
    E = vector_to_skew_symmetric_matrix(t) @ R
    return E

def least_squares_triangulation(K, Rti, Rtj, xi, xj):
    invRi = np.linalg.inv(Rti[:3, :3])
    ti = Rti[:3, 3]
    
    invRj = np.linalg.inv(Rtj[:3, :3])
    tj = Rtj[:3, 3]
    
    invK = np.linalg.inv(K)
    
    num_matches = xi.shape[1]
    print(xi.shape, xj.shape)
    print(invK.shape)
    print(invRi.shape)
    print(invRj.shape)
    x = xi[:, 1]
    y = xj[:, 1]
    
    A = np.hstack([(invRi @ invK @ x).reshape((-1, 1)), (-invRj @ invK @ y).reshape((-1, 1))])
    print(A.shape)
    b = tj - ti
    print(b.shape)
    
    e = np.linalg.pinv(A) @ b
    e = np.linalg.inv(A.T @ A) @ A.T @ b
    print(e)
    estimated_points = np.array([np.linalg.pinv(np.hstack([(invRi @ invK @ xi[:, i]).reshape((-1, 1)), (-invRj @ invK @ xj[:, i]).reshape((-1, 1))])) @ (tj - ti) for i in np.arange(num_matches)]).T
    print(estimated_points.shape)
    print(estimated_points)
    # return estimated_points
    
    # A = np.hstack([invRi @ invK @ xi, -invRj @ invK @ xj])
    # b = tj - ti
    # u_ls = np.linalg.pinv(A) @ b
     
# add a ones block to the end  
def to_homogeneous_coordinates(x):
    return np.vstack([x, np.ones((1, x.shape[1]))])
  
def compute_epipolar_distance(image_plane_dim, epipolar_lines, image_points):
    
    c, r = image_plane_dim # (480, 640)
    reprojection_error = []
    num = epipolar_lines.shape[1]
    for i in np.arange(num):
        line = epipolar_lines[:, i]
        p2 = image_points[:, i] 
        # map line to image plane
        # a vector can be described by two points
        p0 = np.array([0, -line[2]/line[1] ])
        p1 = np.array([c, -(line[2]+line[0]*c)/line[1]])
        
        error = point_line_distance(p0, p1, p2)
        reprojection_error.append(error)
        
    reprojection_error = np.array(reprojection_error)              
    return reprojection_error

# given:
# - a line defined through two points p1 and p2 (2-dim)
# - a point p3 (2-dim)
# find the minimum distance between the point and the line
def point_line_distance(p1, p2, p3): 
        
    num = np.abs(np.cross((p2-p1), (p3-p1)))
    den = np.linalg.norm(p2-p1)
    distance = num / den
    
    return distance

def line_to_line_through_two_points(line):
    # a*x + b*y + c = 0 (general form)
    a, b, c = line
    
    # x = (-by - c) / a
    # y = (-ax - c) / b
    
    # x = 0 ==> y = -c/b
    # y = 0 ==> x = -c/a
    p1 = [0, -c/b]
    p2 = [-c/a, 0]
    return p1, p2
       
def line_to_line_in_image_plane(image_plane_dim, line):
    h, w = image_plane_dim
    a, b, c = line
    
    # p0 = [0, -c/b]
    # p1 = [w, (-a*w-c)/b]
    
    # p0 = [-c/a, 0]
    # p1 = [(-b*h-c)/a, h]
    
    y0 = -c/b
    if y0 < 0: 
        y0 = 0
        x0 = -c/a
        p0 = [x0, y0]
    elif y0 > h:
        y0 = h
        x0 = (-b*h-c)/a
        p0 = [x0, y0]
    else:
        x0 = 0
        p0 = [x0, y0]    
    y1 = (-a*w-c)/b
    if y1 < 0: 
        y1 = 0
        x1 = -c/a
        p1 = [x1, y1]
    elif y1 > h:
        y1 = h
        x1 = (-b*h-c)/a
        p1 = [x1, y1]        
    else:
        x1 = w
        p1 = [x1, y1]
    
        
    return p0, p1
     
# p_hat : estimate
# p     : ground truth
def absolute_scale_estimation_closed_form(p_hat, p):
    assert p_hat.shape == p.shape
    
    num_points = p.shape[0]
    
    mu_p = np.mean(p, axis=0)
    mu_p_hat = np.mean(p_hat, axis=0)
    
    # sigma_p = np.std(p)
    # sigma_p_hat = np.std(p_hat)
    
    sigma_p_hat = 0
    for point in p_hat:
        sigma_p_hat = sigma_p_hat + np.linalg.norm(point - mu_p_hat)
    
    sigma_p_hat = sigma_p_hat / num_points
    
    SIGMA = np.zeros((3, 3))
    for pi, pi_hat in zip(p, p_hat):
        a = (pi - mu_p).reshape(-1, 1)
        b = (pi_hat - mu_p_hat).reshape(-1, 1)
        c = a.dot(b.T)
        
        SIGMA = SIGMA + c
    
    SIGMA = SIGMA / num_points
    
    U, D, Vw = np.linalg.svd(SIGMA)
    if np.linalg.det(U) * np.linalg.det(Vw) < 0:
        W = np.diag([1, 1, -1])
    else:
        W = np.eye(3)
       
    R = U @ W @ Vw
    s = np.trace(np.diag(D) @ W) / sigma_p_hat
     
    t = mu_p - s*R @ mu_p_hat
    return s, R, t


