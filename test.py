import json
import os
import torch
from config import get_config
from torchvision import transforms
from utils_ import get_instance,visual_position_map
import models
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import zoom
def np_resize(orignal_numpy_tensor, target_size):
    # Calculate the zoom factors for each dimension
    zoom_factors = [float(desired) / float(original) 
                    for desired, original in zip(target_size, orignal_numpy_tensor.shape)]
    
    # Resize the array using nearest-neighbor interpolation
    resized_tensor = zoom(orignal_numpy_tensor, zoom_factors, order=0)
    
    return resized_tensor

def diffusion_mask(inputs: np.array, diffusion_distance: int) -> np.array:
    kernel_size = 2 * diffusion_distance + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    
    # Convolve the inputs with the kernel
    diffused = convolve2d(inputs, kernel, mode='same', boundary='symm')
    
    # Convert the diffused values to binary (0 or 1)
    diffused_binary = (diffused > 0).astype(np.float32)
    
    return diffused_binary
# Parse arguments
TEST_CNT=200
import time
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.configs['model']['name'],args.configs['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.save_dir,f"{args.split_name}_{args.save_name}")))
print("load the checkpoint in {}".format(os.path.join(args.save_dir,f"{args.split_name}_{args.save_name}")))
model.eval()
# Create the visualizations directory if it doesn't exist

# visual_dir = os.path.join(args.result_path, 'visual',str(args.configs['model']['depth']))
visual_dir = os.path.join(args.result_path, 'visual','unet')
os.makedirs(visual_dir, exist_ok=True)
# Test the model and save visualizations
with open(os.path.join('./split',f'{args.split_name}.json'),'r') as f:
    test_split=json.load(f)['test'][:TEST_CNT]
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
img_transforms=transforms.Compose(
    [transforms.Resize((args.configs['image_resize'])),
                transforms.ToTensor()])
begin=time.time()
patch_length=int(args.configs["image_resize"][0]/args.configs['patch_size'])
mask=Image.open('./mask.png').resize((patch_length,patch_length),resample=Image.BILINEAR)
mask=np.array(mask)
mask[mask>0]=1
cnt=0

Threshold=0.5
with torch.no_grad():
    for image_name in test_split:
        data = data_dict[image_name]
        img_path=data['vessel_path'][:-3]+'png'
        img=Image.open(img_path).convert('RGB')
        img=img_transforms(img).unsqueeze(0)

        output_img = model(img.to(device)).cpu().squeeze()
        output_img=torch.sigmoid(output_img).numpy()
        # output_img[output_img<Threshold]=0
        output_img[output_img>=Threshold]=1
        output_img=diffusion_mask(output_img,6)
        output_img=output_img*mask
        
        ridge_path = data['ridge_diffusion_path']
        ridge=Image.open(ridge_path)
        ridge=np.array(ridge)
        ridge[ridge>0]=1

        output_img=np_resize(output_img,ridge.shape)

        if np.sum(output_img*ridge)<np.sum(ridge):
            print(np.sum(output_img*ridge)/np.sum(ridge))
            visual_position_map(img_path,output_img,os.path.join(visual_dir,image_name))
            visual_position_map(img_path,ridge,os.path.join(visual_dir,image_name[:-4]+"_r.jpg"))
            unre=np.where(output_img>0.1,np.zeros_like(output_img),np.ones_like(output_img))*ridge
            visual_position_map(img_path,unre,os.path.join(visual_dir,image_name[:-4]+"_un.jpg"))
            cnt+=1
        # for thresh in np.arange(0., 1., 0.1):
        #     cal_output=np.where(output_img>thresh,np.ones_like(output_img),np.zeros_like(output_img))
        #     print("threshold: ",thresh)
        #     print(np.sum(ridge*cal_output)/np.sum(ridge))
print(cnt,TEST_CNT)
end=time.time()
print(f"Finished testing. Time cost {(end-begin)/100:.4f}")
