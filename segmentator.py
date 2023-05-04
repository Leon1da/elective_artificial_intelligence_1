
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import torch 

import matplotlib.pyplot as plt

import numpy as np

from drawer import *

class Segment:
    def __init__(self, id, label_id, label_name, was_fused, score, mask, valid) -> None:
        self.id = id
        self.label_id = label_id
        self.label_name = label_name
        self.was_fused = was_fused
        self.score = score
        self.mask = mask
        self.valid = valid
    
    def __str__(self) -> str:
        return '# id ' + str(self.id) + ' label ' + str(self.label_id) + ' ' + str(self.label_name) + ' score ' + str(self.score) + ' valid ' + str(self.valid)
            

class SegmentationModule:
    def __init__(self) -> None:
        # load Mask2Former fine-tuned on ADE20k semantic segmentation
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")

        # all the classes that the model is capable to detect, classificate, and segment
        self.classes = self.model.config.id2label  
        
        # 1 if the object is used for scale estimation
        # 0 otherwise 
        self.classes_to_segment = {'wall': 0, 'building': 0, 'sky': 0, 'floor': 0, 'tree': 0, 'ceiling': 0, 'road': 0, 'bed': 1, 
            'windowpane': 1, 'grass': 0, 'cabinet': 1, 'sidewalk': 0, 'person': 0, 'earth': 0, 'door': 1, 
            'table': 1, 'mountain': 0, 'plant': 1, 'curtain': 0, 'chair': 1, 'car': 0,'water': 0, 
            'painting': 1, 'sofa': 0, 'shelf': 1, 'house': 0, 'sea': 0, 'mirror': 1, 'rug': 0, 
            'field': 0, 'armchair': 1, 'seat': 1, 'fence': 1, 'desk': 1, 'rock': 0, 'wardrobe': 1, 
            'lamp': 1, 'bathtub': 0, 'railing': 0, 'cushion': 0, 'base': 0, 'box': 1, 'column': 1, 
            'signboard': 1, 'chest of drawers': 1, 'counter': 1, 'sand': 0, 'sink': 1, 'skyscraper': 0, 'fireplace': 1, 
            'refrigerator': 1, 'grandstand': 0, 'path': 0, 'stairs': 1, 'runway': 0, 'case': 1, 'pool table': 1, 
            'pillow': 0, 'screen door': 1, 'stairway': 0, 'river': 0, 'bridge': 0, 'bookcase': 1, 'blind': 0, 
            'coffee table': 1, 'toilet': 1, 'flower': 1, 'book': 0, 'hill': 0, 'bench': 1, 'countertop': 0, 
            'stove': 1, 'palm': 1, 'kitchen island': 1, 'computer': 1, 'swivel chair': 0, 'boat': 0, 'bar': 0, 
            'arcade machine': 0,'hovel': 0, 'bus': 0, 'towel': 0, 'light': 1,'truck': 0,'tower': 0, 'chandelier': 1, 
            'awning': 0, 'streetlight': 1, 'booth': 0, 'television receiver': 1, 'airplane': 0, 'dirt track': 0, 'apparel': 1, 
            'pole': 1, 'land': 0,'bannister': 0, 'escalator':1, 'ottoman': 0, 'bottle': 1, 'buffet': 0, 
            'poster': 1, 'stage': 0, 'van': 0, 'ship': 0, 'fountain': 1, 'conveyer belt': 1, 'canopy': 0, 
            'washer': 1, 'plaything': 0, 'swimming pool': 0, 'stool': 0, 'barrel': 1, 'basket': 0, 'waterfall': 0, 
            'tent': 0, 'bag': 0, 'minibike': 0, 'cradle': 0, 'oven': 0,'ball': 0, 'food': 0, 
            'step': 0, 'tank': 1, 'trade name': 0, 'microwave': 1, 'pot': 0, 'animal': 0, 'bicycle': 0, 
            'lake': 0, 'dishwasher': 1, 'screen': 1, 'blanket': 1, 'sculpture': 1, 'hood': 0, 'sconce': 0, 
            'vase': 1, 'traffic light': 1, 'tray': 0, 'ashcan': 0, 'fan': 0, 'pier': 0, 'crt screen': 0, 
            'plate': 1, 'monitor': 1, 'bulletin board': 0, 'shower': 1, 'radiator': 0, 'glass': 1, 'clock': 1, 'flag': 0}
    
         
        # {
        # 0: 'wall', 1: 'building', 2: 'sky', 3: 'floor', 4: 'tree', 5: 'ceiling', 6: 'road', 7: 'bed ', 
        # 8: 'windowpane', 9: 'grass', 10: 'cabinet', 11: 'sidewalk', 12: 'person', 13: 'earth', 14: 'door', 
        # 15: 'table', 16: 'mountain', 17: 'plant', 18: 'curtain', 19: 'chair', 20: 'car', 21: 'water', 
        # 22: 'painting', 23: 'sofa', 24: 'shelf', 25: 'house', 26: 'sea', 27: 'mirror', 28: 'rug', 
        # 29: 'field', 30: 'armchair', 31: 'seat', 32: 'fence', 33: 'desk', 34: 'rock', 35: 'wardrobe', 
        # 36: 'lamp', 37: 'bathtub', 38: 'railing', 39: 'cushion', 40: 'base', 41: 'box', 42: 'column', 
        # 43: 'signboard', 44: 'chest of drawers', 45: 'counter', 46: 'sand', 47: 'sink', 48: 'skyscraper', 49: 'fireplace', 
        # 50: 'refrigerator', 51: 'grandstand', 52: 'path', 53: 'stairs', 54: 'runway', 55: 'case', 56: 'pool table', 
        # 57: 'pillow', 58: 'screen door', 59: 'stairway', 60: 'river', 61: 'bridge', 62: 'bookcase', 63: 'blind', 
        # 64: 'coffee table', 65: 'toilet', 66: 'flower', 67: 'book', 68: 'hill', 69: 'bench', 70: 'countertop', 
        # 71: 'stove', 72: 'palm', 73: 'kitchen island', 74: 'computer', 75: 'swivel chair', 76: 'boat', 77: 'bar', 
        # 78: 'arcade machine', 79: 'hovel', 80: 'bus', 81: 'towel', 82: 'light', 83: 'truck', 84: 'tower', 85: 'chandelier', 
        # 86: 'awning', 87: 'streetlight', 88: 'booth', 89: 'television receiver', 90: 'airplane', 91: 'dirt track', 92: 'apparel', 
        # 93: 'pole', 94: 'land', 95: 'bannister', 96: 'escalator', 97: 'ottoman', 98: 'bottle', 99: 'buffet', 
        # 100: 'poster', 101: 'stage', 102: 'van', 103: 'ship', 104: 'fountain', 105: 'conveyer belt', 106: 'canopy', 
        # 107: 'washer', 108: 'plaything', 109: 'swimming pool', 110: 'stool', 111: 'barrel', 112: 'basket', 113: 'waterfall', 
        # 114: 'tent', 115: 'bag', 116: 'minibike', 117: 'cradle', 118: 'oven', 119: 'ball', 120: 'food', 
        # 121: 'step', 122: 'tank', 123: 'trade name', 124: 'microwave', 125: 'pot', 126: 'animal', 127: 'bicycle', 
        # 128: 'lake', 129: 'dishwasher', 130: 'screen', 131: 'blanket', 132: 'sculpture', 133: 'hood', 134: 'sconce', 
        # 135: 'vase', 136: 'traffic light', 137: 'tray', 138: 'ashcan', 139: 'fan', 140: 'pier', 141: 'crt screen', 
        # 142: 'plate', 143: 'monitor', 144: 'bulletin board', 145: 'shower', 146: 'radiator', 147: 'glass', 148: 'clock', 149: 'flag'}
        
        
        
        
    def segmentation(self, image):
        # preprocessing image   
        inputs = self.processor(images=image, return_tensors="pt")
        # run segmentation
        with torch.no_grad():
            outputs = self.model(**inputs)
        # postprocessing image
        results = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        
        segmentation = results['segmentation'].numpy()
        segments_info = results["segments_info"]
        
        segments = []
        for segment in segments_info:
            
            id = segment['id']
            label_id = segment['label_id']
            was_fused = segment['was_fused']
            score = segment['score']
            label_name = self.classes[label_id]
            mask = (segmentation == id)
            mask = (mask * 255).astype(np.uint8)
            valid = 0
            if label_name in self.classes_to_segment and self.classes_to_segment[label_name]:
                valid = 1 # if 1 will be used for scale estimation
                
            segment = Segment(id, label_id, label_name, was_fused, score, mask, valid)
            segments.append(segment)
            print(segment)
            
        return segments
    
    def get_valid_segments(self, segments):
        return [segment for segment in segments if segment.valid]
     
    def get_points_inside_segments(self, segments, points):
        shape = points.shape
        mask = np.zeros((shape[0]), dtype=int)
        for segment in segments:
            if segment.valid:
                for i, point in enumerate(points):
                    if segment.mask[point[1], point[0]]:
                        mask[i] = 1
        return mask, points
                
                
            

# from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# # load Mask2Former fine-tuned on ADE20k semantic segmentation
# processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
# model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(images=image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# # model predicts class_queries_logits of shape `(batch_size, num_queries)`
# # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
# class_queries_logits = outputs.class_queries_logits
# masks_queries_logits = outputs.masks_queries_logits

# # you can pass them to processor for postprocessing
# predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)

