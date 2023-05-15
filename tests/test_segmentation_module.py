import sys, os
sys.path.append(os.getcwd())


from dataset import DatasetType, SfMDataset
from segmentator import SegmentationModule

from drawer import Drawer, SegmentationWindow, WindowName

from PIL import Image

def main():
    
    workdir = os.getcwd()
    
    data = SfMDataset(workdir, DatasetType.ICRA)
    # data.load_poses()
    
    segmentation_module = SegmentationModule()
    
    # Init windows (Visualization)
    drawer = Drawer()
    
    segmention_window = SegmentationWindow(WindowName.Segmentation, classes=segmentation_module.classes) 
    drawer.add_window(segmention_window)
    
    images = data.images_fn
    for index, filename_rgb in enumerate(images):
        print("Image", index, filename_rgb)
        
        rgb_img = Image.open(filename_rgb)
        
        segments = segmentation_module.segmentation(rgb_img)
        
        drawer.clear(window_name=WindowName.Segmentation)
        drawer.draw(window_name=WindowName.Segmentation, image=filename_rgb, segments=segments)
        drawer.update()
        
        
        


if __name__ == "__main__":
   main()