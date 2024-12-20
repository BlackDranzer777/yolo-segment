import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO
import os
from segment_anything import SamPredictor, sam_model_registry

# Step 1: Load YOLOv8 Model (Pre-trained on COCO dataset)
print("Loading YOLOv8...")
yolo_model = YOLO("yolov8n.pt")  # Replace 'yolov8n.pt' with 'yolov8s.pt' or 'yolov8m.pt' for better accuracy

# Step 2: Load Meta SAM Model
print("Loading Meta SAM...")
sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")  # Path to SAM checkpoint
predictor = SamPredictor(sam)

# Step 3: Generate Random Deep Colors
def get_random_deep_color():
    """
    Generate a random deep color in BGR format.
    """
    return [random.randint(50, 255) for _ in range(3)]

# Step 4: Directory for Test Images
test_folder = "./nadir_data_set/test/images"  # Path to the test folder
output_folder = "./test_results"  # Folder to save results
os.makedirs(output_folder, exist_ok=True)

# Step 5: Process Each Image in Test Folder
for image_file in os.listdir(test_folder):
    if image_file.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(test_folder, image_file)
        print(f"Processing: {image_file}")

        # Load and preprocess the image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Failed to load image: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Perform Object Detection with YOLOv8
        print("Performing object detection with YOLOv8...")
        results = yolo_model(image_bgr, conf=0.25)  # Confidence threshold set to 25%
        bounding_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())  # Bounding box coordinates
                class_id = int(box.cls[0])  # Class ID
                confidence = box.conf[0]  # Confidence score
                bounding_boxes.append((x1, y1, x2, y2, class_id, confidence))

                # Draw bounding box on the image
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class {class_id}: {confidence:.2f}"
                cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Use YOLOv8 Bounding Boxes as Prompts for Meta SAM
        print("Performing segmentation with Meta SAM...")
        predictor.set_image(image_rgb)  # Set the image for SAM predictor
        mask_overlay = np.zeros_like(image_rgb)

        for idx, (x1, y1, x2, y2, class_id, confidence) in enumerate(bounding_boxes):
            bbox_sam = np.array([[x1, y1], [x2, y2]])  # Convert YOLO bounding box to SAM format

            # Predict segmentation mask
            masks, _, _ = predictor.predict(box=bbox_sam)

            # Apply random color to each mask
            color = get_random_deep_color()
            for mask in masks:
                mask_overlay[mask > 0] = color  # Apply random deep color

            # Combine image and overlay
            combined_image = cv2.addWeighted(image_rgb, 0.7, mask_overlay, 0.3, 0)

            # Display Segmentation Results
            plt.figure(figsize=(10, 10))
            plt.imshow(combined_image)
            plt.axis("off")
            plt.title(f"Segmentation for Object {idx + 1} (Class {class_id})")
            plt.show()

        # Save the YOLOv8 Detection Results
        output_path_yolo = os.path.join(output_folder, f"yolo_output_{image_file}")
        cv2.imwrite(output_path_yolo, cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR))
        print(f"YOLOv8 detection saved to: {output_path_yolo}")

        # Save the SAM Segmentation Results
        output_path_sam = os.path.join(output_folder, f"sam_output_{image_file}")
        cv2.imwrite(output_path_sam, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
        print(f"Meta SAM segmentation saved to: {output_path_sam}")

print("Processing complete!")
