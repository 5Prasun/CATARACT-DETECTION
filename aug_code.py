import cv2
import numpy as np
import os
import random

# Paths to dataset and output directories
input_dir = r"D:/project eye/dataset/processed_images/train/normal"  # Replace with the path to your dataset
output_dir = "aug_dataset(normal)"  # Replace with the desired output path
os.makedirs(output_dir, exist_ok=True)

import cv2
import numpy as np

def resize_and_pad(image, target_width, target_height):
    # Get the original dimensions
    h, w = image.shape[:2]
    
    # Calculate the aspect ratios
    aspect_ratio = w / h
    target_aspect_ratio = target_width / target_height
    
    # Resize the image based on the aspect ratio
    if aspect_ratio > target_aspect_ratio:
        # Width is the constraining dimension
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Height is the constraining dimension
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    # Resize the image to fit within the target dimensions
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create a new image with the target dimensions and fill with a padding color (e.g., black)
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculate padding values
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Place the resized image in the center of the padded image
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    
    return padded_image


# Define augmentation functions
def random_rotation(image, angle_range=(0, 20)):
    angle = np.random.uniform(-angle_range[1], angle_range[1])
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated_image

def random_translation(image, shift_range=0.1):
    h, w = image.shape[:2]
    max_tx = shift_range * w
    max_ty = shift_range * h
    tx = np.random.uniform(-max_tx, max_tx)
    ty = np.random.uniform(-max_ty, max_ty)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return translated_image

def random_zoom(image, zoom_range=(0.8, 1.2)):
    h, w = image.shape[:2]
    scale = np.random.uniform(zoom_range[0], zoom_range[1])
    new_h, new_w = int(h * scale), int(w * scale)
    zoomed_image = cv2.resize(image, (new_w, new_h))

    # Crop or pad to return to original size
    if scale < 1.0:  # Pad if zoomed out
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        zoomed_image = cv2.copyMakeBorder(zoomed_image, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)
    else:  # Crop if zoomed in
        crop_h = (new_h - h) // 2
        crop_w = (new_w - w) // 2
        zoomed_image = zoomed_image[crop_h:crop_h + h, crop_w:crop_w + w]

    return zoomed_image

def random_horizontal_flip(image):
    if np.random.rand() < 0.5:
        return cv2.flip(image, 1)
    return image

# Function to apply all augmentations
def apply_augmentations(image):
    image = random_rotation(image)
    image = random_translation(image)
    image = random_zoom(image)
    image = random_horizontal_flip(image)
    return image

# Process each image in the dataset
num_augmented_per_image = 5  # Number of augmented images to generate per original image

for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Add more extensions if necessary
        # Load the image
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        
        # Generate augmented images
        for i in range(num_augmented_per_image):
            augmented_image_0 = apply_augmentations(image)
            augmented_image=resize_and_pad(augmented_image_0,400,300)
            augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.png"  # Change the extension if needed
            output_path = os.path.join(output_dir, augmented_filename)
            cv2.imwrite(output_path, augmented_image)
            print(f"Saved {output_path}")
