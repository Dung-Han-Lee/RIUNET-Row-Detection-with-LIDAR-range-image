import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import scipy.misc as m

#User define modules
import paths
import config

my_path = os.path.abspath(os.path.dirname(__file__))

class RangeViewDataset(data.Dataset):
    def __init__(self, mode):
        assert mode == 'train' or mode =='val' or mode =='test', \
                "mode must be one of  the following: train, val, test"
        self.mode = mode
        self.transform_compose = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5]) ])

        self.wimg_base  = os.path.join(my_path, paths.base, mode, "weight")
        self.rimg_base  = os.path.join(my_path, paths.base, mode, "range")
        self.iimg_base  = os.path.join(my_path, paths.base, mode, "intensity")
        self.label_base = os.path.join(my_path, paths.base, mode, "label")

        print("during {}, # of inputs = {}, labels = {}".format(\
            mode, len(os.listdir(self.rimg_base)), len(os.listdir(self.label_base)) ))
        assert len(os.listdir(self.rimg_base))==len(os.listdir(self.label_base))
        
    def __len__(self):
        if config.sanity:
            return config.subset
        return len(os.listdir(self.rimg_base)) 

    def __getitem__(self, index):
        wimg_path = os.path.join(self.wimg_base, ("%04d" % index ) + ".npy")
        rimg_path = os.path.join(self.rimg_base, ("%04d" % index ) + ".npy")
        iimg_path = os.path.join(self.iimg_base, ("%04d" % index ) + ".npy")
        lbl_path = os.path.join(self.label_base, ("%04d" % index ) + ".png")

        # Load weight map
        if(self.mode == 'train' or self.mode =='val'):
            wimg = np.load(wimg_path, allow_pickle=True)

        # Concatenate range & intensity image along z-axis
        rimg = np.load(rimg_path, allow_pickle=True)
        iimg = np.load(iimg_path, allow_pickle=True)
        img  = np.stack((rimg, iimg), axis= 0)
        
        # Convert label to binary(float) image
        gray_scale = True
        lbl = m.imread(lbl_path, gray_scale)
        lbl = 1.0 * (lbl > 0)

        # Normalize input and prepare label
        img, lbl = self.transform(img, lbl)

        if(self.mode == 'train' or self.mode =='val'):
            return wimg, img, lbl

        # Test time
        return rimg, img, lbl
        
    def transform(self, img, lbl):
        img = img.astype(float) / 255.0
        img = torch.from_numpy(img).float()
        img = self.transform_compose(img)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl
    