import torch
import numpy as np
import torch.nn.functional as F

import cv2
from PIL import Image
def generate_position_map(ridge_mask,patch_size,save_path=None):
    w,h=ridge_mask.shape
    assert w==h
    assert w%patch_size==0
    kernel = torch.full((1,1,patch_size,patch_size),1/(patch_size*patch_size))
    heatmap_original=F.conv2d(ridge_mask[None,None,:,:],kernel,stride=patch_size)
    heatmap_original=heatmap_original[0,0]
    heatmap_norm = (heatmap_original - heatmap_original.min()) / (heatmap_original.max() - heatmap_original.min()) * 0.8 + 0.2
    heatmap=torch.where(heatmap_original>0,heatmap_norm,heatmap_original)
    heatmap=heatmap.numpy()
    if save_path:
        Image.fromarray((heatmap * 255).astype(np.uint8)).save(save_path)
    return heatmap

def visual_position_map(image_path, position_embedding, save_path=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # convert to 4-channel image

    # Resize position_embedding to match img shape
    heatmap = cv2.resize(position_embedding, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply color map and convert to 4-channel image
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2BGRA)

    # Modify the alpha channel based on the heatmap values
    heatmap_colored[:, :, 3] = np.where(heatmap > 0.01, 127, 0)  # You can adjust the threshold (0.01) if needed

    # Overlay the heatmap on the image using alpha channel for transparency
    output_image = cv2.addWeighted(img, 1.0, heatmap_colored, 0.5, 0)

    if save_path:
        cv2.imwrite(save_path, output_image)

    return output_image
