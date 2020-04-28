#!/usr/bin/env python3
# coding:utf-8 


import tensorflow as tf
from backbone import Backbone


class VGG16(Backbone):
    def __init__(self):
        super().__init__()
        # conv1
        self.conv1_1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2, padding='same')
        # conv2
        self.conv2_1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2, padding='same')
        # conv3
        self.conv3_1 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv3_2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv3_3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPool2D(2, 2, padding='same')
        # conv4
        self.conv4_1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')
        self.conv4_2 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')
        self.conv4_3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')
        self.pool4 = tf.keras.layers.MaxPool2D(2, 2, padding='same')
        # conv5
        self.conv5_1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')
        self.conv5_2 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')
        self.conv5_3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')
        # Difference from original VGG16:
        # 5th maxpool layer has kernel size = 3 and stride = 1
        self.pool5 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        # atrous conv2d for 6th block
        self.conv6_1 = tf.keras.layers.Conv2D(1024, 3, padding='same', dilation_rate=6, activation='relu')
        self.conv6_2 = tf.keras.layers.Conv2D(1024, 1, padding='same', activation='relu')

    def call(self, inputs):
        output1_1 = self.conv1_1(inputs)
        output1_2 = self.conv1_2(output1_1)
        output1 = self.pool1(output1_2)

        output2_1 = self.conv2_1(output1)
        output2_2 = self.conv2_2(output2_1)
        output2 = self.pool2(output2_2)

        output3_1 = self.conv3_1(output2)
        output3_2 = self.conv3_2(output3_1)
        output3_3 = self.conv3_3(output3_2)
        output3 = self.pool3(output3_3)

        output4_1 = self.conv4_1(output3)
        output4_2 = self.conv4_2(output4_1)
        output4_3 = self.conv4_3(output4_2)
        output4 = self.pool4(output4_3)

        output5_1 = self.conv5_1(output4)
        output5_2 = self.conv5_2(output5_1)
        output5_3 = self.conv5_3(output5_2)

        vgg16_conv4 = output5_3

        output5 = self.pool5(output5_3)
        output6_1 = self.conv6_1(output5)
        output6_2 = self.conv6_2(output6_1)

        vgg16_conv7 = output6_2

        return vgg16_conv4, vgg16_conv7

    def create_vgg16_layers(self):
        vgg16_conv4 = [
            # conv1
            self.conv1_1,
            self.conv1_2,
            self.pool1,
            # conv2
            self.conv2_1,
            self.conv2_2,
            self.pool2,
            # conv3
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.pool3
            # conv4
            self.conv4_1,
            self.conv4_2,
            self.conv4_3,
            self.pool4,
            # conv5
            self.conv5_1,
            self.conv5_2,
            self.conv5_3,
        ]

        x = tf.keras.layers.Input(shape=[None, None, 3])
        out = x
        for layer in vgg16_conv4:
            out = layer(out)

        vgg16_conv4 = tf.keras.Model(x, out)

        vgg16_conv7 = [
            self.pool5,
            # atrous conv2d for 6th block
            self.conv6_1,
            self.conv6_2,
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

