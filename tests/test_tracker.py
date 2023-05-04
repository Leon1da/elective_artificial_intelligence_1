import sys
sys.path.append('/home/leonardo/elective_project/')
# sys.path.append('/home/leonardo/elective_project/simulation')

from dataset import DatasetType, SfMDataset
from tracker import *


def main():
    
    workdir = "/mnt/d"
    
    data = SfMDataset(workdir, DatasetType.ICRA)
    data.load_poses()
    
    tracker = SegmentationModule()
    
    drawer = Drawer() 
    drawer.draw_init(tracker.classes)
    
    images = data.images_fn
    for index, image in enumerate(images):
        print("Image", index, image)
        # open image
        image = Image.open(image)  
        
        # if index == 10: break
        segments = tracker.segmentation(image)
        
        drawer.draw_image(image)
        drawer.draw_segments(image, segments)
        drawer.draw_update()
        
        
        


if __name__ == "__main__":
   main()