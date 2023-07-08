import torch
import numpy as np
from scipy.ndimage import convolve
import cv2

def generate_position_map(ridge_mask,patch_size,save_path=None):
    w,h=ridge_mask.shape
    assert w==h
    assert w%patch_size==0
    kernel = np.ones((patch_size, patch_size))
    heatmap_original = convolve(ridge_mask, kernel, mode='constant', cval=0.0)
    heatmap = (heatmap_original - np.min(heatmap_original)) / (np.max(heatmap_original) - np.min(heatmap_original)) * 0.5 + 0.5
    if save_path:
        torch.save(torch.tensor(heatmap), save_path)
    return heatmap

def visual_position_map(image_path, position_embedding, save_path=None):
    img = cv2.imread(image_path)
    heatmap = cv2.resize(position_embedding, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    output_image = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    if save_path:
        cv2.imwrite(save_path, output_image)
    return output_image
