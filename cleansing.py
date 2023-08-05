import json
import os
import torch
from utils_ import generate_position_map
from PIL import Image
import numpy as np
import sys 
sys.path.append('..')
from ROP_diagnoise import generate_ridge,generate_ridge_diffusion,generate_vessel

def generate_possion_map(data_path, image_resize,patch_size):
    # Generate path_image folder
    os.makedirs(os.path.join(data_path,'position_map_gt'),exist_ok=True)
    
    os.system(f"rm -rf {os.path.join(data_path,'position_map_gt')}/*")

    
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_list=json.load(f)
    for  image_name in data_list:
        data=data_list[image_name]
        mask = Image.open(data['ridge_diffusion_path'])
        mask=mask.resize((image_resize,image_resize))
        mask=torch.from_numpy(np.array(mask, np.float32, copy=False))
        mask[mask!=0]=1
        position_save_path=os.path.join(data_path,'position_map_gt',data['image_name'])
        generate_position_map(mask,patch_size,save_path=position_save_path)
        data['pos_embed_gt_path']=position_save_path
        
    with open(os.path.join(data_path,'ridge','annotations.json'),'w') as f:
        json.dump(data_list,f)

if __name__=='__main__':
    from config import get_config
    args=get_config()
    
    # cleansing
    if args.generate_ridge:
        print("begin to generate ridge")
        generate_ridge(args.json_file_dict,args.path_tar)
    if args.generate_diffusion_mask:
        generate_ridge_diffusion(args.path_tar)
        print(f"generate ridge diffusion")
    if args.generate_vessel:
        generate_vessel(data_path=args.path_tar)
        print(f"generate vessel segmentation in {os.path.join(args.path_tar,'vessel_seg')}")
    generate_possion_map(args.path_tar,args.image_size,args.patch_size)