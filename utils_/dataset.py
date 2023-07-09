import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .tools import Fix_RandomRotation
import json
from PIL import Image
import numpy as np

class rop_posembed_dataset(Dataset):
    def __init__(self, data_path, split,image_size):
        with open(os.path.join(data_path, 'ridge',f'{split}.json'), 'r') as f:
            self.annote=json.load(f)
        self.vessel_path=os.path.join(data_path,'vessel_seg')
        self.split=split
        self.resize=(image_size,image_size)
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
    def __getitem__(self, idx):
        data = self.annote[idx]

        # Read the image and mask
        img = Image.open(data['vessel_path']).convert('RGB')
        # img = Image.open(data['image_path']).convert('RGB')
        gt = Image.open(data['pos_heatmap'])
        img=transforms.Resize(self.resize)(img)
        
        if self.split == "train" :
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)
        # Transform mask back to 0,1 tensor
        gt = torch.from_numpy(np.array(gt, np.float32, copy=False)/255)
        img = transforms.ToTensor()(img)
        return img, gt.squeeze(),data['class']

    def __len__(self):
        return len(self.annote)
