import os
import glob
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d
import torch
import numpy as np
from torch.utils import data


class ColorDataLoader:
    '''
    Cut origin image and save clean and noise data
    '''
    def __init__(self, args):
        files = glob.glob(args.train_file_path + '*.bmp')
        files.sort()
        CleanPic = []
        len_data = len(files)
        for i in range(len_data):
            image = Image.open(files[i])
            im = np.float32(np.array(image) / 255.)
            patches = extract_patches_2d(im, (args.patch_size, args.patch_size), max_patches=args.patch_per_image)
            CleanPic.append(patches)
        CleanPic = np.stack(CleanPic, axis=0)
        CleanPic = CleanPic.reshape((-1,) + CleanPic.shape[-3:])
        np.random.shuffle(CleanPic)
        CleanPic = CleanPic.transpose(0, 3, 1, 2)
        CleanPic = torch.from_numpy(CleanPic)
        self.ClearPicLoader = data.DataLoader(CleanPic, batch_size=args.batch_size, shuffle=False)


class GrayDataLoader:
    '''
    Cut origin image and save clean and noise data
    '''
    def __init__(self, args):
        files = glob.glob(args.train_file_path + '*.bmp')
        files.sort()
        CleanPic = []
        len_data = len(files)
        for i in range(len_data):
            color_image = Image.open(files[i])
            gray_image = color_image.convert('L')
            im = np.float32(np.array(gray_image) / 255.)
            patches = extract_patches_2d(im, (args.patch_size, args.patch_size), max_patches=args.patch_per_image)
            CleanPic.append(patches)
        CleanPic = np.stack(CleanPic, axis=0)
        CleanPic = np.expand_dims(CleanPic, axis=-1)
        CleanPic = CleanPic.reshape((-1,) + CleanPic.shape[-3:])
        np.random.shuffle(CleanPic)
        CleanPic = CleanPic.transpose(0, 3, 1, 2)
        CleanPic = torch.from_numpy(CleanPic)
        self.ClearPicLoader = data.DataLoader(CleanPic, batch_size=args.batch_size, shuffle=False)