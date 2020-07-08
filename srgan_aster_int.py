from __future__ import absolute_import
import sys
sys.path.append('./')

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import os.path as osp
import numpy as np
import math
import time

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler

from config import get_args
from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate
from lib.datasets.concatdataset import ConcatDataset
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists

global_args = get_args(sys.argv[1:])

#import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# ASTER arguments 
args = get_args(sys.argv[1:])
# gan arguements
epoch = 0
n_epochs = 11
dataset_name = "ICDAR/testing/ch4_test_word_images_gt"
batch_size = 32
lr = 0.0002
b1 = 0.5
b2 = 0.999
decay_epoch = 100
n_cpu = 8
hr_height = 256
hr_width = 256
channels = 3
sample_interval = 100
checkpoint_interval = 10

class DataInfo(object):  """
  Save the info about the dataset.
  This a code snippet from dataset.py
  """
  def __init__(self, voc_type):
    super(DataInfo, self).__init__()
    self.voc_type = voc_type

    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)

cuda = torch.cuda.is_available()

hr_shape = (hr_height, hr_width)
# Initialize generator and discriminator and aster model
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(channels, hr_shape[0], hr_shape[1]))
dataset_info = DataInfo(args.voc_type)
aster = model = ModelBuilder(arch=args.arch, rec_num_classes=dataset_info.rec_num_classes,
                       sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=max_len,
                       eos=dataset_info.char2id[dataset_info.EOS], STN_ON=args.STN_ON)
#feature_extractor = FeatureExtractor()
# Set feature extractor to inference mode
#feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda(0)
    discriminator = discriminator.cuda(0)
    #feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda(0)
    criterion_content = criterion_content.cuda(0)
    aster = aster.cuda(0)
# load pretrained aster
checkpoint = load_checkpoint(args.resume)
model.load_state_dict(checkpoint['state_dict'])

if epoch != 0:
    # Load pretrained generator and discriminator
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset(dataset_name, hr_shape=hr_shape),
    batch_size=batch_size,
    shuffle=True
)

from datetime import datetime
dt = datetime.now()
timeStamp = dt.strftime('%Y-%m-%d %H:%M:%S')  # format it to a string
time1 = time.time()
print("Training Started at: ", timeStamp)
# ----------
#  Training
# ----------
aster.eval()
for epoch in range(epoch, n_epochs):
    for i, inputs in enumerate(dataloader):
        # Configure model input
        imgs_lr = Variable(inputs["lr"].type(Tensor))
        imgs_hr = Variable(inputs["hr"].type(Tensor))
        labels = Variable(inputs["label"].type(Tensor))
        label_length = Variable(inputs["label_length"].type(Tensor))

        # Adversarial ground truths
        d_out_shape = discriminator.output_shape
        print("The output shape of the discriminator is: ", d_out_shape)
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), d_out_shape[0], d_out_shape[1]))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        gen_hr = generator(imgs_lr)   # Generate a high resolution image from low resolution input
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)    # Adversarial loss
        # Content loss
        """gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())"""
        input_dict = {}
        imgs, label_encs, lengths = inputs
        images = imgs.to(self.device)
        input_dict['images'] = images   #input dicstionary preparation
        input_dict['rec_targets'] = label
        input_dict['rec_lengths'] = label_length

        output_dict = aster(input_dict)     # model output dicstionary
        batch_size = input_dict['images'].size(0)
        total_loss = 0
        loss_dict = {}
        for k, loss in output_dict['losses'].items():
            loss = loss.mean(dim=0, keepdim=True)
            total_loss += self.loss_weights[k] * loss
            loss_dict[k] = loss.item()

        loss_content = total_loss.item()   # GAN content loss
        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN
        loss_G.backward()
        optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()
        # --------------
        #  Log Progress
        # --------------
        time2 = time.time()
        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [Time: %f] [D loss: %f] [G loss: %f]"
            % (epoch + 1, n_epochs, i + 1, len(dataloader), time2 - time1, loss_D.item(), loss_G.item())
        )
        time1 = time2
        """
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)"""

    if checkpoint_interval != -1 and (epoch + 1) % checkpoint_interval == 0:        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % (epoch + 1))
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % (epoch + 1))

torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % (epoch + 2))
torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % (epoch + 2))

print(".......Training has stopped.......")