import json
import os
import torch
from config import get_config
import numpy as np
from torchvision import transforms
from utils_ import get_instance,visual_position_map,visual_points
import models
from PIL import Image
from scipy.ndimage import zoom
# Parse arguments
TEST_CNT=100
import time
args = get_config()

# Init the result file to store the pytorch model and other mid-result
data_path=args.path_tar
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.configs.MODEL.NAME,
                     image_size=args.image_size,
                     patch_size=args.patch_size,
                     embed_dim=64,
                     depth=3,
                     heads=4,
                     mlp_dim=32,
                    #  dropout=0.
                     )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(args.save_name))
print(f"load the checkpoint in {args.save_name}")
model.eval()
# Create the visualizations directory if it doesn't exist

visual_dir = os.path.join(args.result_path, 'visual')
os.makedirs(visual_dir, exist_ok=True)
os.makedirs(os.path.join(args.result_path,'visual_points'),exist_ok=True)
# Test the model and save visualizations
with open(os.path.join(args.data_path,'ridge','test.json'),'r') as f:
    test_data=json.load(f)[:TEST_CNT]
img_transforms=transforms.Compose(
    [transforms.Resize((args.image_size,args.image_size)),
                transforms.ToTensor()])
begin=time.time()
with torch.no_grad():
    for data in test_data:
        img_path=data['vessel_path']
        img_name=data['image_name']
        img=Image.open(img_path).convert('RGB')
        img=img_transforms(img).unsqueeze(0)

        output_img = model(img.to(device)).cpu().squeeze()
        output_img=torch.sigmoid(output_img)
        print(data['image_name'],output_img.max())
        visual_position_map(data['image_path'],output_img.numpy(),os.path.join(visual_dir,data['image_name']))

end=time.time()
print(f"Finished testing. Time cost {(end-begin)/100:.4f}")
