# Fashion Segmentation

> Model : [Mask R-CNN](https://github.com/matterport/Mask_RCNN) - ([pretrain model](https://drive.google.com/file/d/1FVkhA4ys88C0bvUEb2Uh8lu1lQC7iD8m/view?usp=drive_link))
> 
> Dataset : [Deepfashion2](https://github.com/switchablenorms/DeepFashion2)


## 1. Project Overview
This project aims to convert the DeepFashion2 dataset into the COCO dataset format and to perform instance segmentation on the items worn in the images. The DeepFashion2 dataset, a comprehensive fashion dataset, is converted to the COCO format, a widely-used standard for object detection tasks. Instance segmentation is performed using the Mask R-CNN model, which allows us to not only classify the objects in an image but also to pinpoint the exact pixels of each object.

## 2. Methodology
### 2.1 Dataset Conversion
The DeepFashion2 dataset is first converted to the COCO dataset format. This involves extracting labels for each image that denote the category, bounding box, and segmentation mask of each fashion item in the image. This information is used to create a JSON file in the COCO dataset format.

### 2.2 Instance Segmentation
With the dataset converted into the COCO format, instance segmentation is performed using the Mask R-CNN model. Instance segmentation involves distinguishing each object and segmenting the area of each object down to the pixel level. This enables the precise separation and identification of each fashion item in the images.

## 3. Results
Through this project, the DeepFashion2 dataset was successfully converted into the COCO dataset format and instance segmentation was performed on the items in the images. This allows for each fashion item in the image to be precisely identified and located.

> Compare `test_img/` and `Result/`  

## 4. Usage
The code and usage instructions for this project can be found on this GitHub repository. Refer to the project details and code to convert your own dataset into the COCO format and perform instance segmentation.

```
conda create -n F_seg python==3.7
pip install -r requirements.txt
pip install h5py==2.10.0 --force-reinstall
```


