import json
import os
import torch
from utils_ import generate_position_map
from PIL import Image
import numpy as np
import sys 
sys.path.append('..')

def generate_possion_map(data_path, image_resize,patch_size):
    # Generate path_image folder
    os.makedirs(os.path.join(data_path,'position_map_gt'),exist_ok=True)
    
    os.system(f"rm -rf {os.path.join(data_path,'position_map_gt')}/*")

    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_list=json.load(f)
    for  image_name in data_list:
        if 'ridge' not in data_list[image_name]:
            continue
        data=data_list[image_name]
        mask = Image.open(data['ridge_diffusion_path'])
        mask=mask.resize(image_resize)
        mask=torch.from_numpy(np.array(mask, np.float32, copy=False))
        mask[mask!=0]=1
        position_save_path=os.path.join(data_path,'position_map_gt',image_name)
        generate_position_map(mask,patch_size,save_path=position_save_path)
        data['pos_embed_gt_path']=position_save_path
        
    with open(os.path.join(data_path,'annotations.json'),'w') as f:
        json.dump(data_list,f)
def generate_pos_embed_split(data_path,split_name):
    os.makedirs('./split',exist_ok=True)
    with open(os.path.join(data_path,'split',f'{split_name}.json'),'r') as f:
        orginal_split=json.load(f)
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    pos_embed_split={'train':[],'val':[],'test':[]}
    for split in ['train','val','test']:
        for  image_name in orginal_split[split]:
            if 'ridge' not in data_dict[image_name]:
                continue
            pos_embed_split[split].append(image_name)
    with open(os.path.join('./split',f"{(split_name)}.json"),'w') as f:
        json.dump(pos_embed_split,f)
if __name__=='__main__':
    from config import get_config
    args=get_config()
    if args.generate_ridge_diffusion:
        from utils_ import generate_ridge_diffusion
        generate_ridge_diffusion(args.data_path)
    
    # generate_possion_map(args.data_path,args.configs["image_resize"],args.configs["patch_size"])
    # generate_pos_embed_split(args.data_path,args.split_name)