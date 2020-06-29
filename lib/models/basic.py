# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .cpm_vgg16 import cpm_vgg16
from .mobilenetv3 import mobilenetv3
# from .LK import LK
from .LK_mobilenet import LK

def obtain_model(configure, points):
  if configure.arch == 'cpm_vgg16':
    net = cpm_vgg16(configure, points)
  elif configure.arch == 'mobilenetv3':
    net = mobilenetv3(pts_num=points, input_size=112, width_mult=1.0)
  else:
    raise TypeError('Unkonw type : {:}'.format(configure.arch))
  return net

def obtain_LK(configure, lkconfig, points):
  model = obtain_model(configure, points)
  lk_model = LK(model, lkconfig, points)
  return lk_model
