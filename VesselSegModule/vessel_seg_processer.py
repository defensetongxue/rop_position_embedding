# this file will create an interface for the rop_dig
from . import models
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np


class VesselSegProcesser():
    def __init__(self, model_name,
                 resize=(512, 512)):
        self.model = getattr(models, model_name)()
        checkpoint = torch.load(
            './VesselSegModule/checkpoint/best.pth')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()

        self.resize = resize
        # generate mask
        mask = Image.open('./VesselSegModule/mask.png')
        mask = transforms.Resize(self.resize)(mask)
        mask = transforms.ToTensor()(mask)[0]
        self.mask = mask

        self.transforms = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.3968], [0.1980])
            # the mean and std is cal by 12 rop1 samples
            # TODO using more precise score
        ])

    def __call__(self, img,save_path=None):
        # open the image and preprocess
        # img = Image.open(img_path)
        img = self.transforms(img)

        # generate predic vascular with pretrained model
        img = img.unsqueeze(0)  # as batch size 1
        pre = self.model(img.cuda())
        # the input of the 512 is to match the mini-size of vessel model
        pre = transforms.functional.crop(pre, 0, 0, 512,512)
        
        pre=transforms.Resize(self.resize)(pre)
        pre = pre[0, 0, ...]
        predict = torch.sigmoid(pre).cpu().detach()
        # mask
        predict = torch.where(self.mask < 0.1, self.mask, predict)
        if save_path:
            cv2.imwrite(save_path,
                np.uint8(predict.numpy()*255))
        return predict.numpy()
