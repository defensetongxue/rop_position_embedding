import json
import os
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from scipy.ndimage import zoom
import cv2
def visual_mask(image_path, mask,save_path='./tmp.jpg'):
    # Open the image file.
    image = Image.open(image_path).convert("RGBA")  # Convert image to RGBA

    # Create a blue mask.
    mask_np = np.array(mask)
    mask_blue = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)  # 4 for RGBA
    mask_blue[..., 2] = 255  # Set blue channel to maximum
    mask_blue[..., 3] = (mask_np * 127.5).astype(np.uint8)  # Adjust alpha channel according to the mask value

    # Convert mask to an image.
    mask_image = Image.fromarray(mask_blue)

    # Overlay the mask onto the original image.
    composite = Image.alpha_composite(image, mask_image)

    # Convert back to RGB mode (no transparency).
    rgb_image = composite.convert("RGB")

    # Save the image with mask to the specified path.
    rgb_image.save(save_path)
def visual_annotation(image_path, points, save_path='tmp.jpg'):
    """
    image_path: str, path to the image file
    points: np.array, shape (point_number, 2) containing [x, y] coordinates of points
    save_path: str, path where the annotated image will be saved
    """
    # Load the image
    image = cv2.imread(image_path)

    # Check if image loading is successful
    if image is None:
        raise ValueError("Image not found")

    # Ensure points are a numpy array
    points = np.array(points).astype(int)
    
    # Loop over all points and draw them on the image
    for i, (x, y) in enumerate(points):
        # Draw a circle at each point
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # green circle
        
        # Annotate the point index
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(i), (x + 10, y + 10), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)  # red text
    
    # Save the image
    cv2.imwrite(save_path, image)
