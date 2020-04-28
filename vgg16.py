#!/usr/bin/env python3
# coding:utf-8 


import tensorflow as tf
from backbone import Backbone


class VGG16(Backbone):
    def __init__(self):
        super().__init__()
        pass

    def create_vgg16_layers(self):
        vgg16_conv4 = [
            # conv1
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2, padding='same'),
            # conv2
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2, padding='same'),
            # conv3
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2, padding='same'),
            # conv4
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2, padding='same'),
            # conv5
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
        ]

        x = tf.keras.layers.Input(shape=[None, None, 3])
        out = x
        for layer in vgg16_conv4:
            out = layer(out)

        vgg16_conv4 = tf.keras.Model(x, out)

        vgg16_conv7 = [
            # Difference from original VGG16:
            # 5th maxpool layer has kernel size = 3 and stride = 1
            tf.keras.layers.MaxPool2D(3, 1, padding='same'),
            # atrous conv2d for 6th block
            tf.keras.layers.Conv2D(1024, 3, padding='same',
                        dilation_rate=6, activation='relu'),
            tf.keras.layers.Conv2D(1024, 1, padding='same', activation='relu'),
        ]

        x = tf.keras.layers.Input(shape=[None, None, 512])
        out = x
        for layer in vgg16_conv7:
            out = layer(out)

        vgg16_conv7 = tf.keras.Model(x, out)

        return vgg16_conv4, vgg16_conv7


    def create_extra_layers(self):
        """ Create extra layers
            8th to 11th blocks
        """
        extra_layers = [
            # 8th block output shape: B, 512, 10, 10
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(256, 1, activation='relu'),
                tf.keras.layers.Conv2D(512, 3, strides=2, padding='same',
                            activation='relu'),
            ]),
            # 9th block output shape: B, 256, 5, 5
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(128, 1, activation='relu'),
                tf.keras.layers.Conv2D(256, 3, strides=2, padding='same',
                            activation='relu'),
            ]),
            # 10th block output shape: B, 256, 3, 3
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(128, 1, activation='relu'),
                tf.keras.layers.Conv2D(256, 3, activation='relu'),
            ]),
            # 11th block output shape: B, 256, 1, 1
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(128, 1, activation='relu'),
                tf.keras.layers.Conv2D(256, 3, activation='relu'),
            ]),
            # 12th block output shape: B, 256, 1, 1
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(128, 1, activation='relu'),
                tf.keras.layers.Conv2D(256, 4, activation='relu'),
            ])
        ]

        return extra_layers


    def create_conf_head_layers(self, num_classes):
        """ Create layers for classification
        """
        conf_head_layers = [
            tf.keras.layers.Conv2D(4 * num_classes, kernel_size=3, padding='same'),  # for 4th block
            tf.keras.layers.Conv2D(6 * num_classes, kernel_size=3, padding='same'),  # for 7th block
            tf.keras.layers.Conv2D(6 * num_classes, kernel_size=3, padding='same'),  # for 8th block
            tf.keras.layers.Conv2D(6 * num_classes, kernel_size=3, padding='same'),  # for 9th block
            tf.keras.layers.Conv2D(4 * num_classes, kernel_size=3, padding='same'),  # for 10th block
            tf.keras.layers.Conv2D(4 * num_classes, kernel_size=3, padding='same'),  # for 11th block
            tf.keras.layers.Conv2D(4 * num_classes, kernel_size=1)  # for 12th block
        ]

        return conf_head_layers


    def create_loc_head_layers(self):
        """ Create layers for regression
        """
        loc_head_layers = [
            tf.keras.layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
            tf.keras.layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
            tf.keras.layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
            tf.keras.layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
            tf.keras.layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
            tf.keras.layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
            tf.keras.layers.Conv2D(4 * 4, kernel_size=1)
        ]

        return loc_head_layers

