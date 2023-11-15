import os
import sys
import json
import datetime
import numpy as np




#import skimage.draw
 
# Root directory of the project
ROOT_DIR = os.path.abspath("/home/mineslab-ubuntu/Segmentation")
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)
 
# from cocoapi.PythonAPI.pycocotools.coco import COCO

# from cocoapi.PythonAPI.pycocotools import mask as maskUtils
from mrcnn.config import Config
# from mrcnn import utils
from mrcnn.model import MaskRCNN

import skimage
 
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
 
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################
 
class DeepFashion2Config(Config):
    """Configuration for training on DeepFashion2.
    Derives from the base Config class and overrides values specific
    to the DeepFashion2 dataset.
    """
    # Give the configuration a recognizable name
    NAME = "deepfashion2"
 
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
 
    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # Background + category
    
    USE_MINI_MASK = True
 
    train_img_dir = "data/train/image"
    train_json_path = "data/train_json.json"
    valid_img_dir = "data/validation/image"
    valid_json_path = "data/validation_json.json"
    
# ############################################################
# #  Dataset
# ############################################################
# class DeepFashion2Dataset(utils.Dataset):
#     def load_coco(self, image_dir, json_path, class_ids=None,
#                   class_map=None, return_coco=False):
#         """Load the DeepFashion2 dataset.
#         """
 
#         coco = COCO(json_path)
 
#         # Load all classes or a subset?
#         if not class_ids:
#             # All classes
#             class_ids = sorted(coco.getCatIds())
 
#         # All images or a subset?
#         if class_ids:
#             image_ids = []
#             for id in class_ids:
#                 image_ids.extend(list(coco.getImgIds(catIds=[id])))
#             # Remove duplicates
#             image_ids = list(set(image_ids))
#         else:
#             # All images
#             image_ids = list(coco.imgs.keys())
 
#         # Add classes
#         for i in class_ids:
#             self.add_class("deepfashion2", i, coco.loadCats(i)[0]["name"])
 
#         # Add images
#         for i in image_ids:
#             self.add_image(
#                 "deepfashion2", image_id=i,
#                 path=os.path.join(image_dir, coco.imgs[i]['file_name']),
#                 width=coco.imgs[i]["width"],
#                 height=coco.imgs[i]["height"],
#                 annotations=coco.loadAnns(coco.getAnnIds(
#                     imgIds=[i], catIds=class_ids, iscrowd=None)))
#         if return_coco:
#             return coco
        
#     def load_keypoint(self, image_id):
#         """
#         """
#         image_info = self.image_info[image_id]
#         if image_info["source"] != "deepfashion2":
#             return super(DeepFashion2Dataset, self).load_mask(image_id)
 
#         instance_keypoints = []
#         class_ids = []
#         annotations = self.image_info[image_id]["annotations"]
 
#         for annotation in annotations:
#             class_id = self.map_source_class_id(
#                 "deepfashion2.{}".format(annotation['category_id']))
#             if class_id:
#                 keypoint = annotation['keypoints']
 
#                 instance_keypoints.append(keypoint)
#                 class_ids.append(class_id)
 
#         keypoints = np.stack(instance_keypoints, axis=1)
#         class_ids = np.array(class_ids, dtype=np.int32)
#         return keypoints, class_ids
 
#     def load_mask(self, image_id):
#         """Load instance masks for the given image.
#         Different datasets use different ways to store masks. This
#         function converts the different mask format to one format
#         in the form of a bitmap [height, width, instances].
#         Returns:
#         masks: A bool array of shape [height, width, instance count] with
#             one mask per instance.
#         class_ids: a 1D array of class IDs of the instance masks.
#         """
#         # If not a COCO image, delegate to parent class.
#         image_info = self.image_info[image_id]
#         if image_info["source"] != "deepfashion2":
#             return super(DeepFashion2Dataset, self).load_mask(image_id)
 
#         instance_masks = []
#         class_ids = []
#         annotations = self.image_info[image_id]["annotations"]
#         # Build mask of shape [height, width, instance_count] and list
#         # of class IDs that correspond to each channel of the mask.
#         for annotation in annotations:
#             class_id = self.map_source_class_id(
#                 "deepfashion2.{}".format(annotation['category_id']))
#             if class_id:
#                 m = self.annToMask(annotation, image_info["height"],
#                                    image_info["width"])
#                 # Some objects are so small that they're less than 1 pixel area
#                 # and end up rounded out. Skip those objects.
#                 if m.max() < 1:
#                     continue
#                 # Is it a crowd? If so, use a negative class ID.
#                 if annotation['iscrowd']:
#                     # Use negative class ID for crowds
#                     class_id *= -1
#                     # For crowd masks, annToMask() sometimes returns a mask
#                     # smaller than the given dimensions. If so, resize it.
#                     if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
#                         m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
#                 instance_masks.append(m)
#                 class_ids.append(class_id)
 
#         # Pack instance masks into an array
#         if class_ids:
#             mask = np.stack(instance_masks, axis=2).astype(np.bool)
#             class_ids = np.array(class_ids, dtype=np.int32)
#             return mask, class_ids
#         else:
#             # Call super class to return an empty mask
#             return super(DeepFashion2Dataset, self).load_mask(image_id)
        
#     def image_reference(self, image_id):
#         """Return a link to the image in the COCO Website."""
#         super(DeepFashion2Dataset, self).image_reference(image_id)
 
#     # The following two functions are from pycocotools with a few changes.
 
#     def annToRLE(self, ann, height, width):
#         """
#         Convert annotation which can be polygons, uncompressed RLE to RLE.
#         :return: binary mask (numpy 2D array)
#         """
#         segm = ann['segmentation']
#         if isinstance(segm, list):
#             # polygon -- a single object might consist of multiple parts
#             # we merge all parts into one mask rle code
#             rles = maskUtils.frPyObjects(segm, height, width)
#             rle = maskUtils.merge(rles)
#         elif isinstance(segm['counts'], list):
#             # uncompressed RLE
#             rle = maskUtils.frPyObjects(segm, height, width)
#         else:
#             # rle
#             rle = ann['segmentation']
#         return rle
 
#     def annToMask(self, ann, height, width):
#         """
#         Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
#         :return: binary mask (numpy 2D array)
#         """
#         rle = self.annToRLE(ann, height, width)
#         m = maskUtils.decode(rle)
#         return m
    
# def train(model, config):
#     """
#     """
#     dataset_train = DeepFashion2Dataset()
#     dataset_train.load_coco(config.train_img_dir, config.train_json_path)
#     dataset_train.prepare()
 
#     dataset_valid = DeepFashion2Dataset()
#     dataset_valid.load_coco(config.valid_img_dir, config.valid_json_path)
#     dataset_valid.prepare()
 
#     model.train(dataset_train, dataset_valid,
#                 learning_rate=config.LEARNING_RATE,
#                 epochs=30,
#                 layers='3+')
    
############################################################
#  Splash
############################################################
 
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
 
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash
 
import cv2
 
category = {1:"반팔 상의", 2 : "긴팔 상의", 3 : "반팔 아우터", 4 : "긴팔 아우터", 5 : "조끼", 
            6 : "슬링", 7 : "반바지", 8 : "긴 바지", 9 : "스커트", 10 : "반팔 원피스",
            11 : "긴팔 원피스", 12 : "조끼 원피스", 13 : "슬링 원피스"} 

def splash_and_save(image, result, name):
    
    save_dir = "Result"
    mask = result['masks']
    
    exception = []
    if 1 in result['class_ids'] and 2 in result['class_ids']:
        exception.append(2)
    if 7 in result['class_ids'] and 8 in result['class_ids']:
        exception.append(8)
    if 10 in result['class_ids'] and 11 in result['class_ids']:
        exception.append(11)
    
    for i in range(mask.shape[-1]):
        if result['class_ids'][i] in exception:
            continue
        new_img = np.full((image.shape[0], image.shape[1],3),255)
        new_img[np.where(mask[:,:,i])] = image[np.where(mask[:, :, i])]
        
        y_indices, x_indices = np.where(mask[:, :, i])

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        crop_size = 256

        top_left_x = max(0, x_center - crop_size // 2)
        top_left_y = max(0, y_center - crop_size // 2)
        bottom_right_x = min(image.shape[1], top_left_x + crop_size)
        bottom_right_y = min(image.shape[0], top_left_y + crop_size)

        # 이미지 자르기
        cropped_img = new_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        
        file_path = save_dir + f"/{name}{category[result['class_ids'][i]]}.jpg"
        cv2.imwrite(file_path, cropped_img)
        print(f"Save at {file_path}")

 
def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        img_name = (image_path).split('/')[-1].split('.')[0]
        
        # Read image
        #image = skimage.io.imread(args.image)
        image = cv2.imread(image_path)
        print("img shape :", image.shape)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash_and_save(image, r, img_name)
        print([category[i] for i in r['class_ids']], r['scores'], r['masks'].shape)
    
############################################################
#  main
############################################################
 
 
if __name__ == "__main__":
    ROOT_DIR = os.path.abspath("./")
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    import argparse
 
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Match R-CNN for DeepFashion.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()
 
    """
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    """
 
    print("Weights: ", args.weights)
    print("Logs: ", args.logs)
 
 
    # Configurations
    if args.command == "train":
        config = DeepFashion2Config()
    else:
        class InferenceConfig(DeepFashion2Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()
 
    # Create model
    if args.command == "train":
        model = MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
 
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
 
    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
 
    # Train or evaluate
    if args.command == "train":
        train(model, config)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image, video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
        
        

# from fastapi import FastAPI

# app = FastAPI()

# # @app.get("/")
# # def health_check():
# #     return {"ping":"pong"}


# @app.get("/inference")
# def test_handler():
    
#     class InferenceConfig(DeepFashion2Config):
#         # Set batch size to 1 since we'll be running inference on
#         # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#         GPU_COUNT = 1
#         IMAGES_PER_GPU = 1
#     config = InferenceConfig()
#     model = MaskRCNN(mode="inference", config=config, model_dir="log/")
#     model.load_weights("logs/deepfashion220230910T0138/mask_rcnn_deepfashion2_0025.h5", by_name=True)
#     detect_and_color_splash(model, image_path="test5.png")
#     return {"ping":"pong"}

# if __name__ == "__main__":
    
    
#     class InferenceConfig(DeepFashion2Config):
#         # Set batch size to 1 since we'll be running inference on
#         # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#         GPU_COUNT = 1
#         IMAGES_PER_GPU = 1
#     config = InferenceConfig()
#     model = MaskRCNN(mode="inference", config=config, model_dir="log/")
#     model.load_weights("logs/deepfashion220230910T0138/mask_rcnn_deepfashion2_0025.h5", by_name=True)
#     detect_and_color_splash(model, image_path="test5.png")