import torch
import torchvision as tv 
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from PIL import Image
import cv2
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
 
# Root directory of the project
ROOT_DIR = os.path.abspath("/home/mineslab-ubuntu/Segmentation")
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)
 
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools import mask as maskUtils
# from mrcnn.config import Config
# from mrcnn import utils
# from mrcnn.model import MaskRCNN
 
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
 
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################
 
class DeepFashion2Config():
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
 
    train_img_dir = "/home/mineslab-ubuntu/Segmentation/data/train/image"
    train_json_path = "/home/mineslab-ubuntu/Segmentation/data/train_json.json"
    valid_img_dir = "/home/mineslab-ubuntu/Segmentation/data/validation/image"
    valid_json_path = "/home/mineslab-ubuntu/Segmentation/data/validation_json.json"
    
tv.models.segmentation
    
############################################################
#  Dataset
############################################################
class DeepFashion2Dataset():
    def load_coco(self, image_dir, json_path, class_ids=None,
                  class_map=None, return_coco=False):
        """Load the DeepFashion2 dataset.
        """
 
        coco = COCO(json_path)
 
        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())
 
        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())
 
        # Add classes
        for i in class_ids:
            self.add_class("deepfashion2", i, coco.loadCats(i)[0]["name"])
 
        # Add images
        for i in image_ids:
            self.add_image(
                "deepfashion2", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco
        
    def load_keypoint(self, image_id):
        """
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion2":
            return super(DeepFashion2Dataset, self).load_mask(image_id)
 
        instance_keypoints = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
 
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))
            if class_id:
                keypoint = annotation['keypoints']
 
                instance_keypoints.append(keypoint)
                class_ids.append(class_id)
 
        keypoints = np.stack(instance_keypoints, axis=1)
        class_ids = np.array(class_ids, dtype=np.int32)
        return keypoints, class_ids
 
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion2":
            return super(DeepFashion2Dataset, self).load_mask(image_id)
 
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)
 
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(DeepFashion2Dataset, self).load_mask(image_id)
        
    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        super(DeepFashion2Dataset, self).image_reference(image_id)
 
    # The following two functions are from pycocotools with a few changes.
 
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle
 
    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
    
############################################################
#  Splash
###########################################################
 

def color_splash(image, mask, colors=None, alpha=0.5):
    """
    세그멘테이션 마스크를 색칠하여 원본 이미지 위에 오버레이합니다.

    Parameters:
        image (numpy.ndarray): 원본 이미지.
        mask (numpy.ndarray): 세그멘테이션 마스크. 각 클래스에 해당하는 정수로 이루어진 2D 배열.
        colors (list of tuple, optional): 각 클래스에 대한 색상을 지정하는 튜플의 리스트. 기본값은 None으로, 자동으로 색상을 생성합니다.
        alpha (float, optional): 오버레이의 투명도. 기본값은 0.5.

    Returns:
        result (numpy.ndarray): 색칠된 이미지.
    """
    if colors is None:
        # 자동으로 색상 생성
        num_classes = np.max(mask) + 1  # 클래스 개수 (0부터 시작하므로 +1)
        print(num_classes)
        colors = [tuple(np.random.randint(0, 255, 3)) for _ in range(num_classes)]
    
    result = image.copy()
    
    for class_idx in range(len(colors)):
        # 해당 클래스에 해당하는 마스크를 생성
        class_mask = (mask == class_idx)
        
        # 해당 클래스의 색상 선택
        color = colors[class_idx]
        
        # 원본 이미지에 색칠
        result[class_mask] = (1 - alpha) * result[class_mask] + alpha * np.array(color)
    
    return result
    # """Apply color splash effect.
    # image: RGB image [height, width, 3]
    # mask: instance segmentation mask [height, width, instance count]
 
    # Returns result image.
    # """
    # # Make a grayscale copy of the image. The grayscale copy still
    # # has 3 RGB channels, though.
    # gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # # Copy color pixels from the original color image where mask is set
    # if mask.shape[-1] > 0:
    #     # We're treating all instances as one, so collapse the mask into one layer
    #     mask = (np.sum(mask, -1, keepdims=True) >= 1)
    #     splash = np.where(mask, image, gray).astype(np.uint8)
    # else:
    #     splash = gray.astype(np.uint8)
    # return splash
 
 

def detect_and_color_splash(model, args, image_path=None, video_path=None):
    assert image_path or video_path

    transform_list = [transforms.Resize((256, 256)),
                      transforms.ToTensor()]

    composed_transforms = transforms.Compose(transform_list)
    # Image or video?
    if image_path:
        # Run model segmentation
        print("Running on {}".format(args.image))
        # Read image

        img = Image.open(image_path)
        # Convert image to PyTorch tensor
        image_tensor = composed_transforms(img).unsqueeze(0)
        print(image_tensor.size())
        # Ensure the model is in evaluation mode
        model.eval()


        with torch.no_grad():
            # Perform inference
            output = model(image_tensor)
            
            print(output['out'].size())

            # Get predicted class masks
            masks = torch.argmax(output['out'], dim=1).cpu().numpy()[0]
            print("\nmask :", masks.shape, "image :", img.size, np.max(masks))

        # Colorize the segmentation mask (you may need to define this function)
        resize_image = np.array(img.resize((256,256)))
        splash = color_splash(resize_image, masks)
        
        # Save the output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)

    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
 
        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
 
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)
    
    
class DeepFashion2COCODataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        # 이미지 정보 및 어노테이션 정보 불러오기
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # 이미지 불러오기
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path)

        # 마스크 이미지 생성하기
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for ann in anns:
            cat_id = ann['category_id']
            mask = np.maximum(mask, coco.annToMask(ann) * cat_id)

        mask = Image.fromarray(mask, mode="L")
        if img_path.split('/')[-1][:-4] == "000002":
            img.save("iam.png")
            mask.save("mask.png")
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask
    
def data_transforms(img, mask):
    transform_list = [transforms.Resize((256, 256)),
                      transforms.ToTensor()]
    
    mask_transform_list = [transforms.Resize((256, 256)),
                      transforms.ToTensor()]
    d_transforms = transforms.Compose(transform_list)
    mask_transforms = transforms.Compose(mask_transform_list)
    
    img = d_transforms(img)
    mask = mask_transforms(mask)

    return img, mask
