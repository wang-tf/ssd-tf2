#!/usr/bin/env python3
# coding:utf-8
"""
file_name: ssd.py
author: wang-tf
time: 20200430
"""

import os
import numpy as np
import tensorflow as tf

from mobilenet_v2 import MobileNetV2
from other_layer import ExtraLayer, ConfLayer, LocLayer


class SSD(tf.keras.Model):
    """ Class for SSD model
    Attributes:
        num_classes: number of classes
    """

    def __init__(self, num_classes, arch='ssd300'):
        super(SSD, self).__init__()

        self.backbone_layer = MobileNetV2()
        self.num_classes = num_classes
        relu6 = tf.keras.layers.ReLU(6.)

        # self.batch_norm = tf.keras.layers.BatchNormalization(beta_initializer='glorot_uniform', gamma_initializer='glorot_uniform')
        
        self.block8 = ExtraLayer(256, 3, 2, activation=relu6)
        self.block9 = ExtraLayer(128, 3, 2, activation=relu6)
        self.block10 = ExtraLayer(128, 3, 2, activation=relu6)
        self.block11 = ExtraLayer(64, 3, 2, activation=relu6)
        self.extra_layers = [self.block8, self.block9, self.block10, self.block11]

        self.conf1 = ConfLayer(4, num_classes, 3)
        self.conf2 = ConfLayer(6, num_classes, 3)
        self.conf3 = ConfLayer(6, num_classes, 3)
        self.conf4 = ConfLayer(6, num_classes, 3)
        self.conf5 = ConfLayer(4, num_classes, 3)
        self.conf6 = ConfLayer(4, num_classes, 3)
        self.conf_head_layers = [self.conf1, self.conf2, self.conf3, self.conf4, self.conf5, self.conf6]

        self.loc1 = LocLayer(4, 3)
        self.loc2 = LocLayer(6, 3)
        self.loc3 = LocLayer(6, 3)
        self.loc4 = LocLayer(6, 3)
        self.loc5 = LocLayer(4, 3)
        self.loc6 = LocLayer(4, 3)
        self.loc_head_layers = [self.loc1, self.loc2, self.loc3, self.loc4, self.loc5, self.loc6]

    def compute_heads(self, x, idx):
        """ Compute outputs of classification and regression heads
        Args:
            x: the input feature map
            idx: index of the head layer
        Returns:
            conf: output of the idx-th classification head
            loc: output of the idx-th regression head
        """
        conf = self.conf_head_layers[idx](x)
        conf = tf.reshape(conf, [conf.shape[0], -1, self.num_classes])

        loc = self.loc_head_layers[idx](x)
        loc = tf.reshape(loc, [loc.shape[0], -1, 4])

        return conf, loc

    def init(self):
        self.backbone_layer.init()

    def call(self, x):
        """ The forward pass
        Args:
            x: the input image
        Returns:
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        feature_maps = []
        confs = []
        locs = []
        head_idx = 0
        f1, f2 = self.backbone_layer(x)
        # f1 = self.batch_norm(f1)
        feature_maps.extend([f1, f2])

        x = f2
        for layer in self.extra_layers:
            x = layer(x)
            feature_maps.append(x)

        for feature_out in feature_maps:
            conf, loc = self.compute_heads(feature_out, head_idx)
            confs.append(conf)
            locs.append(loc)
            head_idx += 1

        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)

        return confs, locs


def create_ssd(num_classes, arch, pretrained_type,
               checkpoint_dir=None,
               checkpoint_path=None):
    """ Create SSD model and load pretrained weights
    Args:
        num_classes: number of classes
        pretrained_type: type of pretrained weights, can be either 'VGG16' or 'ssd'
        weight_path: path to pretrained weights
    Returns:
        net: the SSD model
    """
    net = SSD(num_classes, arch)
    # net.summary()
    net(tf.random.normal((1, 800, 800, 3)))
    if pretrained_type == 'base':
        # net.init_vgg16()
        net.init()
    elif pretrained_type == 'latest':
        try:
            paths = [os.path.join(checkpoint_dir, path)
                     for path in os.listdir(checkpoint_dir)]
            latest = sorted(paths, key=os.path.getmtime)[-1]
            net.load_weights(latest)
        except AttributeError as e:
            print('Please make sure there is at least one checkpoint at {}'.format(
                checkpoint_dir))
            print('The model will be loaded from base weights.')
            net.init_vgg16()
        except ValueError as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    latest, arch))
        except Exception as e:
            print(e)
            raise ValueError('Please check if checkpoint_dir is specified')
    elif pretrained_type == 'specified':
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                'Not a valid checkpoint file: {}'.format(checkpoint_path))

        try:
            net.load_weights(checkpoint_path)
        except Exception as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    checkpoint_path, arch))
    else:
        raise ValueError('Unknown pretrained type: {}'.format(pretrained_type))
    return net

