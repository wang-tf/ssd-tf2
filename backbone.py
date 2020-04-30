#!/usr/bin/env python3
# coding:utf-8

import abc
import tensorflow as tf


class Backbone(tf.keras.Models):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def call():
        return NotImplemented
    
