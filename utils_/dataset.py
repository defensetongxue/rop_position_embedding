import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .tools import Fix_RandomRotation
import json
from PIL import Image
import numpy as np

class rop_posembed_dataset(Dataset):
    def __init__(self, data_path, split,split_name,image_resize):
        with open(os.path.join( './split',f'{split_name}.json'), 'r') as f:
            split_list=json.load(f)
        with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
            self.data_list=json.load(f)
        self.split_list=split_list[split]
        self.split=split
        self.resize=transforms.Resize((image_resize))
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
    def __getitem__(self, idx):
        image_name = self.split_list[idx]
        data=self.data_list[image_name]
        # Read the image and mask
        img = Image.open(data['vessel_path']).convert('RGB')
        gt = Image.open(data['pos_embed_gt_path'])
        img=self.resize(img)
        
        if self.split == "train" :
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)
        # Transform mask back to 0,1 tensor
        gt = torch.from_numpy(np.array(gt, np.float32, copy=False)/255)
        img = transforms.ToTensor()(img)
        return img, gt.squeeze(),data['image_path']

    def __len__(self):
        return len(self.split_list)
