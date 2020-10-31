import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from patch_extractor import *
from PIL import Image, ImageEnhance
from torchvision.transforms import transforms
#---------- setting -----------#
LABELS = ['Normal', 'Benign', 'InSitu', 'Invasive']
IMAGE_SIZE = (2048, 1536)
PATCH_SIZE = 256
#---------- setting -----------#
class PathWiseDataset(Dataset):

    def __init__(self, path, stride=PATCH_SIZE, rotate=False, flip=False, enhance=False):
        super().__init__()
        
        wp = int((IMAGE_SIZE[0] - PATCH_SIZE) / stride + 1)
        hp = int((IMAGE_SIZE[1] - PATCH_SIZE) / stride +1)
        
        labels = {name: index for index in range(len(LABELS)) for name in glob.glob(path + '/' + LABELS[index] + '/*.tif')}
        self.path = path
        self.stride = stride
        self.labels = labels
        self.names = list(sorted(labels.keys()))
        self.shape = (len(labels), wp, hp, (4 if rotate else 1), (2 if flip else 1), (2 if enhance else 1))
        self.augment_size = np.prod(self.shape) / len(labels)
    def __getitem__(self, index):

        im, xpatch, ypatch, rotation, flip, enhance = np.unravel_index(index, self.shape)
        with Image.open(self.names[im]) as img:
            extractor = PatchExtractor(img=img, patch_size=PATCH_SIZE, stride=self.stride)
            patch = extractor.extract_patch((xpatch, ypatch))
            
            if rotation != 0:
                patch = patch.rotate(rotation * 90)
            
            if flip != 0:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
            
            if enhance != 0:
                factors = np.random.uniform(.5, 1.5, 3)
                patch = ImageEnhance.Color(patch).enhance(factors[0])
                patch = ImageEnhance.Contrast(patch).enhance(factors[1])
                patch = ImageEnhance.Brightness(patch).enhance(factors[2])
            
            label = self.labels[self.names[im]]
            return transforms.ToTensor()(patch), label

    def __len__(self):
        return np.prod(self.shape)

class TestDataset(Dataset):
    def __init__(self, path, stride=PATCH_SIZE, augment=False):
        super().__init__()
        if os.path.isdir(path):
            names = [name for name in glob.glob(path + '/*.tif')]
        else:
            names = [path]
        self.path = path
        self.stride = stride
        self.augment = augment
        self.names = list(sorted(names))

    def __getitem__(self, index):
        file = self.names[index]
        with Image.open(file) as img:
            bins = 8 if self.augment else 1
            extractor = PatchExtractor(img=img, patch_size=PATCH_SIZE, stride=self.stride)
            b = torch.zeros((bins, extractor.shape()[0] * extractor.shape()[1], 3, PATCH_SIZE, PATCH_SIZE))
            for k in range(bins):
                if k % 4 != 0:
                    img = img.rotate((k % 4) * 90)
                if k // 4 != 0:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                extractor = PatchExtractor(img=img, patch_size=PATCH_SIZE, stride=self.stride)
                patches = extractor.extract_patches()
                for i in range(len(patches)):
                    b[k, i] = transforms.ToTensor()(patches[i])
            return b, file

    def __len__(self):
        return len(self.names)

if __name__ == '__main__':
    test_path = './data/test'
    if os.path.isdir(test_path):
        names = [name for name in glob.glob(test_path + '/*.tif')]
    else:
        names = [test_path]
    print(names)
    names = list(sorted(names))
    print(names)
    print(names[0])
