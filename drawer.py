import pyviz3d.visualizer as viz
import numpy as np
import os 
        
class BrowserDrawer:
    def __init__(self) -> None:
        self.visualizer = viz.Visualizer()
    
    def draw_points(self, points, color=np.array([[0.], [0.], [0.]]), point_size=0.1, name='points'):
        if color.shape[1] == 1: 
            color = np.full(points.shape, color[:, 0])
            
        self.visualizer.add_points(name=name, positions=points, colors=color, point_size=point_size)
    
    
    
    def draw_frames(self, frames, colors, sizes, names):
        for frame, color, size, name in zip(frames, colors, sizes, names):
            self.draw_frame(frame, color, size, name)
        
    
    def draw_frame(self, frame, color=None, size=0.1, name='frame'):
        dx, dy, dz = frame[:3, :3]
        origin = frame[:3, 3]
        
        scale = 0.1
        
        lines_start = np.array([origin, origin, origin])
        lines_end = np.array([origin + dx*scale,
                     origin + dy*scale,
                     origin + dz*scale])
        if color: colors = np.array([color, color, color])
        else: colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]) #rgb
        
        self.visualizer.add_lines(name, lines_start, lines_end, colors=colors)
       
    def draw_lines(self, starts, ends, color=np.array([[0.], [0.], [0.]]), name='lines'):
        assert starts.shape == ends.shape
        if color.shape[1] == 1: 
            color = np.full(starts.shape, color[:, 0])
        
        self.visualizer.add_lines(name, starts, ends, colors=color) 
    
    def show(self):
        self.visualizer.save('Plots')
        
        cmd = 'python3 -m http.server 6008'
        os.system(cmd)
 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import enum

class WindowType(enum.Enum):
   Plot3D = 1
   Plot2D = 2

class WindowName(enum.Enum):
    ScaleStatistics = 'ScaleStatistics'
    Segmentation = 'Segmentation'
    Points = 'Points'
    Poses = 'Poses'

      

class DrawerWindow:
    
    def __init__(self, window_name: WindowName):
        self.window_name = window_name
    
    def update(self):
        self.fig.canvas.draw_idle() # draw on the same image
        plt.pause(0.001)
    
    def clear(self):
        for scatter in self.scatters:
            scatter.remove() # remove previous plotted points
        self.scatters = []
        
            
        
class ScaleEstimationWindow(DrawerWindow):
    
    def __init__(self, window_name: WindowName):
        fig, ax = plt.subplots(2,4)
        
        self.fig = fig
        self.ax = ax
        
        super().__init__(window_name)
    
    def draw(self, **kwargs):
        
        # SICP
        if 'sicp_transformation_evolution' in kwargs:
            transformation_evolution = kwargs['sicp_transformation_evolution']
            num_iterations = len(transformation_evolution)
            scale = [T[3, 3] for T in transformation_evolution]
                
            lines = self.ax[0, 0].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_iteration = x_data[-1]
                last_scale = y_data[-1]
                iterations = list(range(last_iteration + 1, last_iteration + num_iterations + 1))
                iterations.insert(0, last_iteration) 
                scale.insert(0, last_scale)
            else:
                iterations = list(range(num_iterations))
            
            # print(iterations)
            # print(scale)
            self.ax[0, 0].plot(iterations, scale, c='b')
            self.ax[0, 0].set_title('sicp_transformation_evolution')
        
        if 'sicp_error' in kwargs:
            error = kwargs['sicp_error']
            lines = self.ax[0, 1].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_iteration = x_data[-1]
                last_error = y_data[-1]
                iterations = [last_iteration, last_iteration + 1]
                error = [last_error, error]
            else:
                iterations = [0]
                error = [error]
                
            # print(iterations)
            # print(error)

            self.ax[0, 1].plot(iterations, error, c='b')
            self.ax[0, 1].set_title('sicp_error')
     
        if 'sicp_chi_evolution' in kwargs:
            chi_evolution = kwargs['sicp_chi_evolution']
            num_iterations = len(chi_evolution)
                
            lines = self.ax[0, 2].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_iteration = x_data[-1]
                last_chi = y_data[-1]
                iterations = list(range(last_iteration + 1, last_iteration + num_iterations + 1))
                iterations.insert(0, last_iteration) 
                chi_evolution.insert(0, last_chi)
            else:
                iterations = list(range(num_iterations))
            
            # print(iterations)
            # print(chi_evolution)
            self.ax[0, 2].plot(iterations, chi_evolution, c='b')
            self.ax[0, 2].set_title('sicp_chi_evolution')
        
        if 'sicp_inliers_evolution' in kwargs:
            inliers_evolution = kwargs['sicp_inliers_evolution']
            num_iterations = len(inliers_evolution)
                
            lines = self.ax[0, 3].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_iteration = x_data[-1]
                last_inliers = y_data[-1]
                iterations = list(range(last_iteration + 1, last_iteration + num_iterations + 1))
                iterations.insert(0, last_iteration) 
                inliers_evolution.insert(0, last_inliers)
            else:
                iterations = list(range(num_iterations))
            
            # print(iterations)
            # print(inliers_evolution)
            self.ax[0, 3].plot(iterations, inliers_evolution, c='b')
            self.ax[0, 3].set_title('sicp_inliers_evolution')
        
        
        # MICP
        if 'micp_transformation_evolution' in kwargs:
            transformation_evolution = kwargs['micp_transformation_evolution']
            num_iterations = len(transformation_evolution)
            scale = [T[3, 3] for T in transformation_evolution]
                
            lines = self.ax[1, 0].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_iteration = x_data[-1]
                last_scale = y_data[-1]
                iterations = list(range(last_iteration + 1, last_iteration + num_iterations + 1))
                iterations.insert(0, last_iteration) 
                scale.insert(0, last_scale)
            else:
                iterations = list(range(num_iterations))
            
            # print(iterations)
            # print(scale)
            self.ax[1, 0].plot(iterations, scale, c='b')
            self.ax[1, 0].set_title('micp_transformation_evolution')
        
        if 'micp_error' in kwargs:
            error = kwargs['micp_error']
            lines = self.ax[1, 1].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_iteration = x_data[-1]
                last_error = y_data[-1]
                iterations = [last_iteration, last_iteration + 1]
                error = [last_error, error]
            else:
                iterations = [0]
                error = [error]
            
            # print(iterations)
            # print(error)
            self.ax[1, 1].plot(iterations, error, c='b')
            self.ax[1, 1].set_title('micp_error')
            
        if 'micp_chi_evolution' in kwargs:
            chi_evolution = kwargs['micp_chi_evolution']
            num_iterations = len(chi_evolution)
                
            lines = self.ax[1, 2].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_iteration = x_data[-1]
                last_chi = y_data[-1]
                iterations = list(range(last_iteration + 1, last_iteration + num_iterations + 1))
                iterations.insert(0, last_iteration) 
                chi_evolution.insert(0, last_chi)
            else:
                iterations = list(range(num_iterations))
            
            # print(iterations)
            # print(chi_evolution)
            self.ax[1, 2].plot(iterations, chi_evolution, c='b')
            self.ax[1, 2].set_title('micp_chi_evolution')
        
        if 'micp_inliers_evolution' in kwargs:
            inliers_evolution = kwargs['micp_inliers_evolution']
            num_iterations = len(inliers_evolution)
                
            lines = self.ax[1, 3].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_iteration = x_data[-1]
                last_inliers = y_data[-1]
                iterations = list(range(last_iteration + 1, last_iteration + num_iterations + 1))
                iterations.insert(0, last_iteration) 
                inliers_evolution.insert(0, last_inliers)
            else:
                iterations = list(range(num_iterations))
            
            # print(iterations)
            # print(inliers_evolution)
            self.ax[1, 3].plot(iterations, inliers_evolution, c='b')
            self.ax[1, 3].set_title('micp_inliers_evolution')
        
         
        
            
    
    
class SegmentationWindow(DrawerWindow):
    
    def __init__(self, window_name: WindowName, classes):
        fig, ax = plt.subplots(1,1)
        scatter = ax.scatter([], [])
        
        self.fig = fig
        self.ax = ax
        self.scatters = []
        
        num_classes = len(classes.items())
        cmap = plt.cm.get_cmap('hsv', num_classes) # color map (1 color for each possible class)
        
        self.patches = {}
        for key in classes.keys():
            label_name = classes[key]
            color = cmap(key)
            alpha = 0.5 # set opacity
            color = (color[0], color[1], color[2], alpha)
            
            self.patches[key] = mpatches.Patch(color=color, label=label_name)
        
        # self.patches[-1] = mpatches.Patch(color=(1, 1, 1, 0.9), label='Invalid')
        self.handles = []
        
        super().__init__(window_name)
        
    def draw(self, **kwargs):
            
        if 'image' in kwargs and 'segments' in kwargs:
            image = kwargs['image']
            segments = kwargs['segments']
            self.draw_image_and_segments(image, segments)
        
        if 'keypoints' in kwargs and 'mask' in kwargs:
            keypoints = kwargs['keypoints']
            mask = kwargs['mask']
            self.draw_keypoints(keypoints, mask)    
         
    def draw_image_and_segments(self, image, segments):
        # draw image
        image = plt.imread(image)
        self.ax.imshow(image)
        
        # set a color for each segment in the mask
        h, w, _ = image.shape
        color_mask = np.zeros((h, w, 4))
        for segment in segments:
            label_id = segment.label_id
            mask = segment.mask
            
            if segment.valid:
                patch = self.patches[label_id]
            else:
                # setup INVALID patch
                label = segment.label_name + " [INVALID]"
                patch = mpatches.Patch(color=(1, 1, 1, 0.3), label=label)
                # patch = self.patches[-1]
                
            self.handles.append(patch)
            color = patch.get_edgecolor()
            color_mask[mask==255] = color
            
                       
        # draw segments 
        self.ax.imshow(color_mask)
        
    def draw_keypoints(self, keypoints, mask=None):
        
        total_keypoints = len(mask)
        total_segmented_keypoints = np.sum(mask)
        print("keypoints", total_keypoints)
        print("segmented keypoints", total_segmented_keypoints)
        
        # self.scatter.remove() # remove previous scatter plot
        # #ff0000 (red) segmented
        # #000000 (black) outliers (not segmented)
        colors = ['#ff0000' if m else '#000000' for m in mask]
        xs = keypoints[:, 0]
        ys = keypoints[:, 1]
        scatter = self.ax.scatter(xs, ys, s=0.1, c=colors)
        self.scatters.append(scatter)
   
   
class PointsWindow(DrawerWindow):
    def __init__(self, window_name: WindowName):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='3d')
        
        self.fig = fig
        self.ax = ax
        self.scatters = []
        
        super().__init__(window_name)
        
    def draw(self, **kwargs):
        if 'points' in kwargs:
            points = kwargs['points']
        if 'color' in kwargs:
            color = kwargs['color']
        self.plot_points(points, color)
    
    def plot_points(self, points, color=None):
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        scatter = self.ax.scatter(xs, ys, zs, marker='.', c=color, linewidths=0.2)
        self.scatters.append(scatter)

class PosesWindow(DrawerWindow):
    
    def __init__(self, window_name: WindowName):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='3d')
        
        self.fig = fig
        self.ax = ax
        self.scatters = []
        
        super().__init__(window_name)
        
    def draw(self, **kwargs):
        if 'tvecs' in kwargs:
            tvecs = kwargs['tvecs']
        if 'color' in kwargs:
            color = kwargs['color']
        self.plot_tvecs(tvecs, color)
        
    def plot_tvecs(self, tvecs, color):
        xs = tvecs[:, 0]
        ys = tvecs[:, 1]
        zs = tvecs[:, 2]
        scatter = self.ax.scatter(xs, ys, zs, marker='.', c=color, linewidths=0.2)
        self.scatters.append(scatter)
    

    
class Drawer:
    
    def __init__(self, windows={}) -> None:
        self.windows = windows
    
    def add_window(self, window: DrawerWindow):
        self.windows[window.window_name] = window
    
    def draw(self, window_name: WindowName, **kwargs):
        self.windows[window_name].draw(**kwargs)
    
    def update(self):
        for window_key in self.windows:
            self.windows[window_key].update()
            
    def clear(self, window_name: WindowName):
        self.windows[window_name].clear()
        
