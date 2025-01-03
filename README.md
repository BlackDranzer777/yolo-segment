# yolo-segment

Download "sam_vit_h_4b8939.pth" from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file


# YOLO + Meta SAM Segmentation Pipeline

## Overview
This repository contains code for object detection and segmentation using YOLOv8 and Meta SAM (Segment Anything Model). The pipeline uses YOLOv8 for object detection and provides bounding boxes as prompts to Meta SAM for precise segmentation.

---

## Requirements

### Install Dependencies
Make sure you have Python 3.8+ installed. Install the required packages using:
```bash
pip install -r requirements.txt
```

# Dataset Setup

## Roboflow Dataset
This repository includes a pre-configured dataset exported from Roboflow in the YOLOv8 format.

### Training Dataset
- The dataset is stored in `nadir_data_set/` with subfolders for `train`, `valid`, and `test`.

### Dataset YAML File
Ensure the `data.yaml` file is correctly pointing to the dataset structure.

#### Example of `data.yaml`:
```yaml
train: ./nadir_data_set/train/images
val: ./nadir_data_set/valid/images
test: ./nadir_data_set/test/images

nc: 4  # Number of classes
names: ['trees', 'cars', 'houses', 'street-lamps']
```

# Training the Model
### Using YOLOv8
Install ultralytics:
```bash
pip install ultralytics
```


### Train YOLOv8 on the dataset:
```bash
yolo task=detect mode=train data=./nadir_data_set/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```
The trained weights will be saved in the runs/detect/train/weights/ directory.

# Running the Pipeline
### YOLO + SAM Pipeline
The pipeline detects objects using YOLOv8 and refines segmentation using Meta SAM.

Run the script:

```bash
python final_main.py
```
Input: Place test images in nadir_data_set/test/images.
Output: Results will be saved in test_results/


# Directory Structure
```bash
.
├── final_main.py
├── final_main_just_sam.py
├── nadir_data_set/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── sam_vit_h_4b8939.pth
├── yolov8n.pt
└── README.md
```
