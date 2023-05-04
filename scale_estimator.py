import numpy as np
from utils.geometric_utils import *

class ScaleEstimatorModule:
    def __init__(
        self, 
        linear_transformation_guess=np.eye(4), 
        similarity_transformation_guess=np.eye(4)) -> None:
        
        self.linear_transformation_guess = linear_transformation_guess
        self.similarity_transformation_guess = similarity_transformation_guess
        
        # Configuration
        self.iterations = 0
        self.dumping = 0
        self.kernel_threshold = 0
    
        # Statistics
        self.linear_chi_evolution = []
        self.linear_num_inliers_evolution = []
        self.linear_transformation_evolution = []
        
        self.similarity_chi_evolution = []
        self.similarity_num_inliers_evolution = []
        self.similarity_transformation_evolution = []
        
    def configure(self, iterations, dumping, kernel_threshold):
        self.iterations = iterations
        self.dumping = dumping
        self.kernel_threshold = kernel_threshold
    
    def reset(self):
        self.linear_transformation_guess = np.eye(4)
        self.similarity_transformation_guess = np.eye(4)
        
        # Configuration
        self.iterations = 0
        self.dumping = 0
        self.kernel_threshold = 0
        
        
    def recover_similarity(self, points, measurements):
        
        similarity_guess = self.similarity_transformation_guess
        iterations = self.iterations
        dumping = self.dumping
        kernel_threshold = self.kernel_threshold
        
        similarity, chi_stats, inliers, similarity_evolution = self.robust_sicp(
            similarity_guess=similarity_guess, 
            points=points,
            measurements=measurements,
            iterations=iterations,
            dumping=dumping,
            kernel_threshold=kernel_threshold)
        self.similarity_transformation_guess = similarity
        return similarity, chi_stats, inliers, similarity_evolution
    
    def recover_linear_transformation(self, points, measurements):
        linear_transformation_guess = self.linear_transformation_guess
        iterations = self.iterations
        dumping = self.dumping
        kernel_threshold = self.kernel_threshold
        
        lienar_transformation, chi_stats, inliers, lienar_transformation_evolution = self.robust_micp(
            linear_transformation_guess=linear_transformation_guess, 
            points=points,
            measurements=measurements,
            iterations=iterations,
            dumping=dumping,
            kernel_threshold=kernel_threshold)

        self.linear_transformation_guess = lienar_transformation
        return lienar_transformation, chi_stats, inliers, lienar_transformation_evolution
    
    def robust_sicp(self, similarity_guess, points, measurements, iterations, dumping, kernel_threshold):
        X = similarity_guess
        chi_stats = np.zeros(iterations)
        num_inliers = np.zeros(iterations)
        X_evolution = np.zeros((iterations, 4, 4))
       
        for iteration in np.arange(iterations):
            H = np.zeros((7, 7))
            b = np.zeros((7, 1))
            X_evolution[iteration, :, :] = X
            for index, point in enumerate(points):
                measurement = measurements[index]
                e, J = error_and_jacobian_sicp(X, point, measurement)
                chi = e.T @ e
                if chi > kernel_threshold:
                    # print('chi > kernel_threshold', chi, '>', kernel_threshold)
                    e = e * np.sqrt(kernel_threshold / chi)
                    chi = kernel_threshold
                else:
                    # print('chi < kernel_threshold', chi, '<', kernel_threshold)
                    num_inliers[iteration] = num_inliers[iteration] + 1
                chi_stats[iteration] = chi_stats[iteration] + chi
                H = H + J.T @ J
                b = b + J.T @ e
            # print(X)
                
            if not np.linalg.det(H):
                print("Singular Matrix")
                return X, chi_stats, num_inliers, X_evolution
            H = H + np.eye(7) * dumping
            dx = -np.linalg.solve(H, b)
            X = v2s(dx) @ X
        return X, chi_stats, num_inliers, X_evolution

    def robust_micp(self, linear_transformation_guess, points, measurements, iterations, dumping, kernel_threshold):
        X = linear_transformation_guess
        chi_stats = np.zeros(iterations)
        num_inliers = np.zeros(iterations)
        X_evolution = np.zeros((iterations, 4, 4))
        
        for iteration in np.arange(iterations):
            H = np.zeros((6, 6))
            b = np.zeros((6, 1))
            X_evolution[iteration, :, :] = X
            
            for index, point in enumerate(points):
                measurement = measurements[index]
                e, J = error_and_jacobian_micp(X, point, measurement)
                chi = e.T @ e
                if chi > kernel_threshold:
                    # print('chi > kernel_threshold', chi, '>', kernel_threshold)
                    e = e * np.sqrt(kernel_threshold / chi)
                    chi = kernel_threshold
                else:
                    # print('chi < kernel_threshold', chi, '<', kernel_threshold)
                    num_inliers[iteration] = num_inliers[iteration] + 1
                chi_stats[iteration] = chi_stats[iteration] + chi
                
                H = H + J.T @ J
                b = b + J.T @ e
            # print(X)
            if not np.linalg.det(H):
                print("Singular Matrix")
                return X, chi_stats, num_inliers, X_evolution
            H = H + np.eye(6) * dumping
            dx = -np.linalg.solve(H, b)
            X = v2t(dx) @ X
        return X, chi_stats, num_inliers, X_evolution

    

def error_and_jacobian_sicp(X, p, z):
    t = X[:3, 3]
    R = X[:3, :3]
    s = X[3, 3]    
    
    z_hat = s * (R @ p + t) # prediction
    e = z_hat - z
    
    J = np.zeros((3, 7))
    J[:3, :3] = np.eye(3)
    J[:3, 3:6] = -skew(z_hat)
    J[:3, 6] = z_hat
    return e.reshape(-1, 1), J

def error_and_jacobian_micp(X, p, z):
    t = X[:3, 3]
    R = X[:3, :3]
    
    z_hat = R @ p + t # prediction
    e = z_hat - z
    J = np.zeros((3, 6))
    J[:3, :3] = np.eye(3)
    J[:3, 3:] = -skew(z_hat)
    return e.reshape(-1, 1), J


def sicp(similarity_guess, points, measurements, iterations):
    X = similarity_guess
    chi_stats = np.zeros(iterations)
    for iteration in np.arange(iterations):
        print(X)
        H = np.zeros((7, 7))
        b = np.zeros((7, 1))
        chi = 0
        for index, point in enumerate(points):
            measurement = measurements[index]
            e, J = error_and_jacobian_sicp(X, point, measurement)
            H = H + J.T @ J
            b = b + J.T @ e
            chi = chi + e.T @ e
        
        if not np.linalg.det(H):
            print("Singular Matrix")
            return X, chi_stats
        print(X)
        
        dx = -np.linalg.solve(H, b)
        X = v2s(dx) @ X
        
        chi_stats[iteration] = chi
        
    return X, chi_stats

def micp(linear_transformation_guess, points, measurements, iterations):
    X = linear_transformation_guess
    chi_stats = np.zeros(iterations)
    for iteration in np.arange(iterations):
        H = np.zeros((6, 6))
        b = np.zeros((6, 1))
        chi = 0
        for index, point in enumerate(points):
            measurement = measurements[index]
            e, J = error_and_jacobian_micp(X, point, measurement)
            H = H + J.T @ J
            b = b + J.T @ e
            chi = chi + e.T @ e
        
        print(X)
        
        if not np.linalg.det(H):
            print("Singular Matrix")
            return X, chi_stats
        
        dx = -np.linalg.solve(H, b)
        X = v2t(dx) @ X
        chi_stats[iteration] = chi
        
    return X, chi_stats
            
    
    