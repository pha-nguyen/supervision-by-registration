# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# from .cpm_vgg16 import cpm_vgg16
from .pfld import PFLDInference
from .LK import LK
import torch

def obtain_model(configure, points):
  if configure.arch == 'pfld':
    net = PFLDInference(points)
  else:
    raise TypeError('Unkonw type : {:}'.format(configure.arch))
  return net

def obtain_LK(configure, lkconfig, points):
  model = obtain_model(configure, points)

  checkpoint = torch.load("/home/ubuntu/checkpoint_epoch_969.pth.tar")
  model.load_state_dict(checkpoint["state_dict"], strict=False)
  lk_model = LK(model, lkconfig, points)
  return lk_model
