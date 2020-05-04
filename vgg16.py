#!/usr/bin/env python3
# coding:utf-8 

import numpy as np
import tensorflow as tf
from backbone import Backbone


class VGG16(Backbone):
    def __init__(self):
        super().__init__()
        # block1
        self.block1_conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv1')
        self.block1_conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv2')
        self.block1_pool = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='block1_pool')
        # block2
        self.block2_conv1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv1')
        self.block2_conv2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv2')
        self.block2_pool = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='block2_pool')
        # block3
        self.block3_conv1 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv1')
        self.block3_conv2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv2')
        self.block3_conv3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv3')
        self.block3_pool = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='block3_pool')
        # block4
        self.block4_conv1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv1')
        self.block4_conv2 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv2')
        self.block4_conv3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv3')
        self.block4_pool = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='block4_pool')
        # block5
        self.block5_conv1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv1')
        self.block5_conv2 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv2')
        self.block5_conv3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv3')

        # Difference from original VGG16:
        # 5th maxpool layer has kernel size = 3 and stride = 1
        self.block5_pool = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='block5_pool')
        # atrous conv2d for 6th block
        self.block6_conv1 = tf.keras.layers.Conv2D(1024, 3, padding='same', dilation_rate=6, activation='relu', name='block6_conv1')
        self.block6_conv2 = tf.keras.layers.Conv2D(1024, 1, padding='same', activation='relu', name='block6_conv2')

    def call(self, inputs):
        output1_1 = self.block1_conv1(inputs)
        output1_2 = self.block1_conv2(output1_1)
        output1 = self.block1_pool(output1_2)

        output2_1 = self.block2_conv1(output1)
        output2_2 = self.block2_conv2(output2_1)
        output2 = self.block2_pool(output2_2)

        output3_1 = self.block3_conv1(output2)
        output3_2 = self.block3_conv2(output3_1)
        output3_3 = self.block3_conv3(output3_2)
        output3 = self.block3_pool(output3_3)

        output4_1 = self.block4_conv1(output3)
        output4_2 = self.block4_conv2(output4_1)
        output4_3 = self.block4_conv3(output4_2)
        output4 = self.block4_pool(output4_3)

        output5_1 = self.block5_conv1(output4)
        output5_2 = self.block5_conv2(output5_1)
        output5_3 = self.block5_conv3(output5_2)

        output5 = self.block5_pool(output5_3)
        output6_1 = self.block6_conv1(output5)
        output6_2 = self.block6_conv2(output6_1)

        return output4_3, output6_2 

    def init(self):
        """ Initialize the VGG16 layers from pretrained weights
        and the rest from scratch using xavier initializer
        """
        origin_vgg = tf.keras.applications.VGG16(weights='imagenet')
        len_vgg16_conv4 = 17
        weights = []
        for i in range(len_vgg16_conv4):
            w_and_b = origin_vgg.get_layer(index=i+1).get_weights()
            weights.extend(w_and_b)

        fc1_weights, fc1_biases = origin_vgg.get_layer(index=-3).get_weights()
        fc2_weights, fc2_biases = origin_vgg.get_layer(index=-2).get_weights()
        conv6_weights = np.random.choice(np.reshape(fc1_weights, (-1)), (3, 3, 512, 1024))
        conv6_biases = np.random.choice(fc1_biases, (1024, ))
        conv7_weights = np.random.choice(np.reshape(fc2_weights, (-1)), (1, 1, 1024, 1024))
        conv7_biases = np.random.choice(fc2_biases, (1024, ))
        weights.extend([conv6_weights, conv6_biases, conv7_weights, conv7_biases])
        self.set_weights(weights)


class ExtraLayer(tf.keras.layers.Layer):
    def __init__(self, units, kernel_size, strides):
        super().__init__()
        padding = 'same' if strides !=1 else 'valid'
        self.conv1 = tf.keras.layers.Conv2D(units, 1, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(units*2, kernel_size, strides=strides, padding=padding, activation='relu')

    def call(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        return output


class ConfLayer(tf.keras.layers.Layer):
    def __init__(self, num_anchor, num_classes, kernel_size):
        super().__init__()
        padding = 'same' if kernel_size != 1 else 'valid'

        self.conv1 = tf.keras.layers.Conv2D(num_anchor * num_classes, kernel_size=kernel_size, padding=padding)

    def call(self, inputs):
        output = self.conv1(inputs)
        return output


class LocLayer(tf.keras.layers.Layer):
    def __init__(self, num_anchor, kernel_size):
        super().__init__()
        padding = 'same' if kernel_size != 1 else 'valid'

        self.conv1 = tf.keras.layers.Conv2D(num_anchor * 4, kernel_size=kernel_size, padding=padding)

    def call(self, inputs):
        output = self.conv1(inputs)
        return output

