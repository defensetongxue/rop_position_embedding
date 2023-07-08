import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .tools import Fix_RandomRotation
import json
from PIL import Image
import numpy as np

class rop_posembed_dataset(Dataset):
    def __init__(self, data_path, split):
        with open(os.path.join(data_path, 'ridge',f'{split}.json'), 'r') as f:
            self.annote=json.load(f)
        self.split=split
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
    def __getitem__(self, idx):
        data = self.annote[idx]

        # Read the image and mask
        img = Image.open(data['vessel_seg']).convert('RGB')
        gt = Image.open(data['mask_path'])
        img=transforms.Resize((224,224))(img)
        gt=torch.load(data['position_save_path'])
        
        if self.split == "train" :
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)
        # Transform mask back to 0,1 tensor
        gt = torch.from_numpy(np.array(gt, np.float32, copy=False))
        img = transforms.ToTensor()(img)
        
        return img, gt.squeeze(),data['class']

    def __len__(self):
        return len(self.annote)
