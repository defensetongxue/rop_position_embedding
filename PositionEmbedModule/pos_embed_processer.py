# this file will create an interface for the rop_dig
from . import models
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np


class PosEmbedProcesser():
    def __init__(self, model_name,
                 vessel_resize,image_orignal_size,patch_size):
        self.model = getattr(models, model_name)(
                    patch_size=patch_size,
                    image_size=vessel_resize,
                    embed_dim=64,
                     depth=3,
                     heads=4,
                     mlp_dim=32,
                    #  dropout=0.
                     )
        checkpoint = torch.load(
            './PositionEmbedModule/checkpoint/pos_embed.pth')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()

        self.patch_size=patch_size
        # generate mask
        self.image_original_size=image_orignal_size
        self.transforms = transforms.Compose([
            transforms.Resize((vessel_resize,vessel_resize))
        ])

    def __call__(self, vessel,save_path=None):
        # open the image and preprocess
        vessel = self.transforms(vessel)

        # generate predic vascular with pretrained model
        img = img.unsqueeze(0)  # as batch size 1
        pre = self.model(img.cuda())
        predict = torch.sigmoid(pre).cpu().detach()
        predict=F.interpolate(predict.unsqueeze(0),
                              size=self.image_original_size,
                              mode='nearest')
        predict.unsqueeze(0)

        if save_path:
            cv2.imwrite(save_path,
                np.uint8(predict.numpy()*255))
        return predict.numpy()
