import numpy as np
import cv2
from PIL import Image
from scipy.signal import convolve2d
def generate_position_map(ridge_mask_path,patch_length,diffusion_size=4,save_path=None):
    ridge_mask=Image.open(ridge_mask_path)
    ridge_mask=ridge_mask.resize(size=(patch_length,patch_length),  resample=Image.BILINEAR)
    ridge_mask=np.array(ridge_mask)
    
    kernel_size=2*diffusion_size+1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    
    # Convolve the inputs with the kernel
    ridge_mask = convolve2d(ridge_mask, kernel, mode='same', boundary='symm')
    
    ridge_mask[ridge_mask>0]=1
    if save_path:
        Image.fromarray((ridge_mask * 255).astype(np.uint8)).save(save_path)
    return ridge_mask

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
