""" will create the dataset for GAN_ASTER model"""
from __future__ import absolute_import
import glob
import random
import os
import numpy as np

import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import os
import pickle
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np
import random
import sys
import six

from torch.utils import data
from torch.utils.data import sampler

from lib.utils.labelmaps import get_vocabulary, labels2strs
from lib.utils import to_numpy

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

"""
    will create dicstionary from the gt.txt key is:img_file_name and value is: gt text
"""
def img_recog(gt_path):
    f = open(gt_path, 'r')

    ff = f.readlines()
    print("The data:\n",ff)
    img_recog_dics = {}
    for files in ff:
        x = files.split()
        #print(x)
        img_recog_dics[x[0][:-1]] = x[1][1:-1]
        f.close()
    #print("the dics is", img_recog_dics)
    return img_recog_dics

"""
    will create dataset for the gan and aster combined
"""
class ImageDataset(Dataset):
    def __init__(self, root, gt_path, hr_shape, reso_factor, max_len, voc_type ):
        hr_height, hr_width = hr_shape
        #Transforms for low resolution images and high resolution image
        self.lr_transform = transforms.Compose( [ transforms.Resize((hr_height //reso_factor, hr_height // reso_factor), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean, std),])                 
        self.hr_transform = transforms.Compose( [ transforms.Resize((32, 100), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean, std),])
        self.files = sorted(glob.glob(root + "/*.*"))
        self.img_recog_dics = img_recog(gt_path)
        #self.img_recog_dics = {"demo.png": "available"}
        self.voc_type = voc_type
        self.max_len = max_len

        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)
        self.lowercase = (voc_type == 'LOWERCASE')

    def __getitem__(self, index):
        file_name = self.files[index % len(self.files)]
        img = Image.open(file_name)
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        rec_text = self.img_recog_dics[file_name.split('/')[-1]]
        # reconition labels
        word = rec_text
        if self.lowercase:
            word = word.lower()
        ## fill with the padding token
        label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
        label_list = []
        for char in word:
            if char in self.char2id:
                label_list.append(self.char2id[char])
            else:
                ## add the unknown token
                print('{0} is out of vocabulary.'.format(char))
                label_list.append(self.char2id[self.UNKNOWN])
        ## add a stop token
        label_list = label_list + [self.char2id[self.EOS]]
        assert len(label_list) <= self.max_len
        label[:len(label_list)] = np.array(label_list)
        # label length
        label_length = len(label_list)
        return {"lr": img_lr, "hr": img_hr, "label": label, "label_length": label_length }
    def __len__(self):                                                                                                                                                                                                                  def __len__(self):
        return len(self.files)