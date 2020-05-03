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

# from layers import create_vgg16_layers, create_extra_layers, create_conf_head_layers, create_loc_head_layers
from vgg16 import VGG16, ExtraLayer, ConfLayer, LocLayer


class SSD(tf.keras.Model):
    """ Class for SSD model
    Attributes:
        num_classes: number of classes
    """

    def __init__(self, num_classes, arch='ssd300'):
        super(SSD, self).__init__()

        self.backbone_layer = VGG16()
        self.num_classes = num_classes

        self.batch_norm = tf.keras.layers.BatchNormalization(beta_initializer='glorot_uniform', gamma_initializer='glorot_uniform')
        
        # self.extra_layers = self.backbone_layer.create_extra_layers()
        # self.l8, self.l9, self.l10, self.l11, self.l12 = self.extra_layers
        self.block8 = ExtraLayer(256, 3, 2)
        self.block9 = ExtraLayer(128, 3, 2)
        self.block10 = ExtraLayer(128, 3, 1)
        self.block11 = ExtraLayer(128, 3, 1)
        self.block12 = ExtraLayer(128, 4, 1)
        self.extra_layers = [self.block8, self.block9, self.block10, self.block11, self.block12]

        # self.conf_head_layers = self.backbone_layer.create_conf_head_layers(num_classes)
        self.conf1 = ConfLayer(4, num_classes, 3)
        self.conf2 = ConfLayer(6, num_classes, 3)
        self.conf3 = ConfLayer(6, num_classes, 3)
        self.conf4 = ConfLayer(6, num_classes, 3)
        self.conf5 = ConfLayer(4, num_classes, 3)
        self.conf6 = ConfLayer(4, num_classes, 3)
        self.conf7 = ConfLayer(4, num_classes, 1)
        self.conf_head_layers = [self.conf1, self.conf2, self.conf3, self.conf4, self.conf5, self.conf6, self.conf7]

        # self.loc_head_layers = self.backbone_layer.create_loc_head_layers()
        self.loc1 = LocLayer(4, 3)
        self.loc2 = LocLayer(6, 3)
        self.loc3 = LocLayer(6, 3)
        self.loc4 = LocLayer(6, 3)
        self.loc5 = LocLayer(4, 3)
        self.loc6 = LocLayer(4, 3)
        self.loc7 = LocLayer(4, 1)
        self.loc_head_layers = [self.loc1, self.loc2, self.loc3, self.loc4, self.loc5, self.loc6, self.loc7]

        if arch == 'ssd300':
            self.extra_layers.pop(-1)
            self.conf_head_layers.pop(-2)
            self.loc_head_layers.pop(-2)

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

    def init_vgg16(self):
        """ Initialize the VGG16 layers from pretrained weights
            and the rest from scratch using xavier initializer
        """
        origin_vgg = tf.keras.applications.VGG16(weights='imagenet')
        len_vgg16_conv4 = 17
        weights = []
        for i in range(len_vgg16_conv4):
            init_w = origin_vgg.get_layer(index=i+1).get_weights()
            weights.extend(init_w)

        fc1_weights, fc1_biases = origin_vgg.get_layer(index=-3).get_weights()
        fc2_weights, fc2_biases = origin_vgg.get_layer(index=-2).get_weights()

        conv6_weights = np.random.choice(np.reshape(fc1_weights, (-1,)), (3, 3, 512, 1024))
        conv6_biases = np.random.choice(fc1_biases, (1024,))

        conv7_weights = np.random.choice(np.reshape(fc2_weights, (-1,)), (1, 1, 1024, 1024))
        conv7_biases = np.random.choice(fc2_biases, (1024,))
        weights.extend([conv6_weights, conv6_biases, conv7_weights, conv7_biases])
        self.backbone_layer.set_weights(weights)

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
        f1 = self.batch_norm(f1)
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
    net(tf.random.normal((1, 512, 512, 3)))
    if pretrained_type == 'base':
        net.init_vgg16()
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

