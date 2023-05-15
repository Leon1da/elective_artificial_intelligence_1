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
    
    
    
    def draw_frames(self, frames, colors=None, sizes=None, names='frames'):
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
    
    def show(self, name=None):
        fn = 'Plots'
        if name:
            fn = 'Plots/' + name
        self.visualizer.save(fn)
        
        
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
    PosesComplete = 'PosesComplete'
    PosesEvaluation = 'PosesEvaluation'

      

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
        # fig, ax = plt.subplots(1, 2)
        
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
            # lines = self.ax[0, 1].get_lines()
            lines = self.ax[0, 0].get_lines()
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
            # lines = self.ax[1, 1].get_lines()
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
        
        # GENERAL
        if 'absolute_error_iterations' in kwargs:
            absolute_error_iterations = kwargs['absolute_error_iterations']
            ate, iteration = absolute_error_iterations[0], absolute_error_iterations[1]
            print(ate, iteration)
                
            lines = self.ax[0, 0].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_iteration = x_data[-1]
                last_ate = y_data[-1]
                iteration_ = [last_iteration, iteration]
                ate_ = [last_ate, ate]
            else:
                iteration_ = [iteration]
                ate_ = [ate]
            
            self.ax[0, 0].plot(iteration_, ate_, c='b', marker='.', markersize=4, linewidth=1)
            self.ax[0, 0].set_title('absolute_error_iterations')
        
        if 'absolute_error_keypoints' in kwargs: 
            
            absolute_error_keypoints = kwargs['absolute_error_keypoints']
            ate, keypoints_number = absolute_error_keypoints[0], absolute_error_keypoints[1]
            print(ate, keypoints_number)
                
            lines = self.ax[0, 3].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_keypoints_number = x_data[-1]
                last_ate = y_data[-1]
                keypoints_number_ = [last_keypoints_number, keypoints_number]
                ate_ = [last_ate, ate]
            else:
                keypoints_number_ = [keypoints_number]
                ate_ = [ate]
            
            self.ax[0, 3].plot(keypoints_number_, ate_, c='b', marker='.', markersize=4, linewidth=1)
            self.ax[0, 3].set_title('absolute_error_keyframes')
        
        if 'absolute_error_keyframes' in kwargs:
            absolute_error_keyframes = kwargs['absolute_error_keyframes']
            ate, keyframe_number = absolute_error_keyframes[0], absolute_error_keyframes[1]
            print(ate, keyframe_number)
                
            lines = self.ax[0, 2].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_keyframe_number = x_data[-1]
                last_ate = y_data[-1]
                keyframe_number_ = [last_keyframe_number, keyframe_number]
                ate_ = [last_ate, ate]
            else:
                keyframe_number_ = [keyframe_number]
                ate_ = [ate]
            
            self.ax[0, 2].plot(keyframe_number_, ate_, c='b', marker='.', markersize=4, linewidth=1)
            self.ax[0, 2].set_xlabel('# keyframes')
            self.ax[0, 2].set_ylabel('ATE [m]')
            
            self.ax[0, 2].set_title('absolute_error_keyframes')
        
        if 'percentage_error_trajectory_length' in kwargs:
            percentage_error_trajectory_length = kwargs['percentage_error_trajectory_length']
            percentage_ate, trajectory_length = percentage_error_trajectory_length[0], percentage_error_trajectory_length[1]
            print(percentage_ate, trajectory_length)
                
            lines = self.ax[0, 1].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_trajectory_length = x_data[-1]
                last_percentage_ate = y_data[-1]
                trajectory_length_ = [last_trajectory_length, trajectory_length]
                percentage_ate_ = [last_percentage_ate, percentage_ate]
            else:
                trajectory_length_ = [trajectory_length]
                percentage_ate_ = [percentage_ate]
            
            self.ax[0, 1].plot(trajectory_length_, percentage_ate_, c='b', marker='.', markersize=4, linewidth=1)
            self.ax[0, 1].set_ylim([0, 100])
            self.ax[0, 1].set_ylabel('translation error [%]')
            self.ax[0, 1].set_xlabel('trajectory length [m]')
            # self.ax[0, 1].set_title('percentage_error_trajectory_length')

        if 'scale_keyframes' in kwargs:
            scale_keyframes = kwargs['scale_keyframes']
            scale, keyframes = scale_keyframes[0], scale_keyframes[1]
            print(scale, keyframes)
                
            lines = self.ax[1, 0].get_lines()
            num_lines = len(lines)
            if num_lines:
                x_data, y_data = lines[-1].get_data()  
                last_keyframes = x_data[-1]
                last_scale = y_data[-1]
                
                scale_ = [last_scale, scale]
                keyframes_ = [last_keyframes, keyframes]
            else:
                scale_ = [scale]
                keyframes_ = [keyframes]
            
            self.ax[1, 0].plot(keyframes_, scale_, c='b', marker='.', markersize=4, linewidth=1)
            self.ax[1, 0].set_xlabel('# keyframe')
            self.ax[1, 0].set_ylabel('scale')
            # self.ax[1, 0].set_title('scale_keyframes')
        
        if 'keypoints' in kwargs:
            keypoints = kwargs['keypoints']
            keyframes, total_keypoints, segmentated_keypoints, optimization_keypoints = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
            
            bins = [keyframes, keyframes + 1]
            
            x = [total_keypoints]
            self.ax[1, 1].stairs(x, bins, lw=1, ec="#000000", fc="#29339B", alpha=0.5, fill=True)
            
            x = [segmentated_keypoints]
            self.ax[1, 1].stairs(x, bins, lw=1, ec="#000000", fc="#348AA7", alpha=0.5, fill=True)

            x = [optimization_keypoints]            
            self.ax[1, 1].stairs(x, bins, lw=1, ec="#000000", fc="#5DD39E", alpha=0.5, fill=True)
            
            self.ax[1, 1].set_xlabel('# keyframe')
            self.ax[1, 1].set_ylabel('# keypoints')
            
            c1 = mpatches.Patch(color='#29339B', label='segmented keypoints')
            c2 = mpatches.Patch(color='#348AA7', label='total keypoints')
            c3 = mpatches.Patch(color='#5DD39E', label='otimization keypoints')
            self.ax[1, 1].legend(handles=[c1, c2, c3])
            
    
    
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
        
        self.handles = []
        
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
        self.ax.legend(handles = self.handles)
        
    def draw_keypoints(self, keypoints, mask=None):
        
        # total_keypoints = len(mask)
        # total_segmented_keypoints = np.sum(mask)
        # print("keypoints", total_keypoints)
        # print("segmented keypoints", total_segmented_keypoints)
        
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
        ax1 = fig.add_subplot(221, projection='3d') # 3d plot
        ax1.set_title('poses')
        
        ax2 = fig.add_subplot(222) # xy plot
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('y [m]')
        
        ax3 = fig.add_subplot(223) # xz plot
        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('z [m]')
        
        ax4 =fig.add_subplot(224) # yz plot
        ax4.set_xlabel('y [m]')
        ax4.set_ylabel('z [m]')
        
    
        
        self.fig = fig
        self.ax = [ax1, ax2, ax3, ax4]
        
        self.plots = []
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
        
        scatter = self.ax[0].scatter(xs, ys, zs, marker='.', c=color, s=0.2)
        p1 = self.ax[1].plot(xs, ys, c=color)[0]
        p2 = self.ax[2].plot(xs, zs, c=color)[0]
        p3 = self.ax[3].plot(ys, zs, c=color)[0]
        
        self.plots = [p1, p2, p3]
        self.scatters.append(scatter)
     
            
    def clear(self):
        for plot in self.plots:
            plot.remove()

        super().clear()
        
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
        
