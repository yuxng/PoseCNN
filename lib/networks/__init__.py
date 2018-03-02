# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

from .vgg16 import vgg16
from .vgg16_convs import vgg16_convs
from .vgg16_full import vgg16_full
from .vgg16_det import vgg16_det
from .vgg16_gan import vgg16_gan
from .dcgan import dcgan
from .resnet50 import resnet50
from .fcn8_vgg import fcn8_vgg
from . import factory
