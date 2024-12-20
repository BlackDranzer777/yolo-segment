import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Step 1: Load Meta SAM Model
print("Loading Meta SAM...")
sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")  # Path to SAM checkpoint
mask_generator = SamAutomaticMaskGenerator(sam)

# Step 2: Directory for Test Images
test_folder = "./nadir_data_set/test/images"  # Path to the test folder
output_folder = "./test_results"  # Folder to save results
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# Step 3: Process Each Image in Test Folder
for image_file in os.listdir(test_folder):
    if image_file.endswith((".jpg", ".png", ".jpeg")):  # Process image files only
        image_path = os.path.join(test_folder, image_file)
        print(f"Processing: {image_file}")

        # Load and preprocess the image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Failed to load image: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Generate masks with SAM
        print("Generating masks...")
        masks = mask_generator.generate(image_rgb)

        # Overlay masks on the original image
        mask_overlay = np.zeros_like(image_rgb)
        for mask in masks:
            segmentation = mask["segmentation"]
            mask_overlay[segmentation > 0] = [255, 0, 0]  # Red overlay

        combined_image = cv2.addWeighted(image_rgb, 0.7, mask_overlay, 0.3, 0)

        # Save and display the results
        output_path = os.path.join(output_folder, f"sam_output_{image_file}")
        cv2.imwrite(output_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
        print(f"Saved segmentation result to: {output_path}")

        # Optional: Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(combined_image)
        plt.axis("off")
        plt.title(f"Meta SAM Segmentation - {image_file}")
        plt.show()

print("Processing complete!")
