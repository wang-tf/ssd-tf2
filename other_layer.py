#!/usr/bin/env python3
# coding:utf-8 

import numpy as np
import tensorflow as tf


class ExtraLayer(tf.keras.layers.Layer):
    def __init__(self, units, kernel_size, strides, activation='relu'):
        super().__init__()
        padding = 'same' if strides !=1 else 'valid'
        self.conv1 = tf.keras.layers.Conv2D(units, 1, activation=activation)
        self.conv2 = tf.keras.layers.Conv2D(units*2, kernel_size, strides=strides, padding=padding, activation=activation)

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

