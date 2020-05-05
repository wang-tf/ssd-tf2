#!/usr/bin/env python3
# encoding:utf-8

"""
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

import tensorflow as tf
from backbone import Backbone

# Define ReLU6 activation
relu6 = tf.keras.layers.ReLU(6.)


def _conv_block(inputs, filters, kernel, strides):
  """Convolution Block
  This function defines a 2D convolution operation with BN and relu6.

  # Arguments
    inputs: Tensor, input tensor of conv layer.
    filters: Integer, the dimensionality of the output space.
    kernel: An integer or tuple/list of 2 integers, specifying the
        width and height of the 2D convolution window.
    strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
        Can be a single integer to specify the same value for
        all spatial dimensions.

  # Returns
    Output tensor.
  """
  x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  return relu6(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
  """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
  """

  tchannel = inputs.shape[-1] * t

  x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

  x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = relu6(x)

  x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  if r:
    x = tf.keras.layers.add([x, inputs])
  return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def MobileNetV2(input_shape, k, plot_model=False):
    """MobileNetv2
    This function defines a MobileNetv2 architecture.

    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        plot_model: Boolean, whether to plot model architecture or not
    # Returns
        MobileNetv2 model.
    """

    inputs = tf.keras.layers.Input(shape=input_shape, name='input')
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    # x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)  # ssd output
    x = _bottleneck(x, 160, (3, 3), 6, 2)
    for i in range(1, n):
        x = _bottleneck(x, 160, (3, 3), 6, 1, True)

    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 1280))(x)
    x = tf.keras.layers.Dropout(0.3, name='Dropout')(x)
    x = tf.keras.layers.Conv2D(k, (1, 1), padding='same')(x)
    x = tf.keras.layers.Activation('softmax', name='final_activation')(x)
    output = tf.keras.layers.Reshape((k,), name='output')(x)
    model = tf.keras.models.Model(inputs, output)
    model.summary()
    if plot_model:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    return model


class MobileNetV2(Backbone):
  def __init__(self):
    super().__init__()
    self.block1 = ConvBlock(32, (3, 3), strides=(2, 2))

    self.block2 = InvertedResidualBlock(16, (3, 3), t=1, strides=1, n=1)
    self.block3 = InvertedResidualBlock(24, (3, 3), t=6, strides=2, n=2)
    self.block4 = InvertedResidualBlock(32, (3, 3), t=6, strides=2, n=3)
    self.block5 = InvertedResidualBlock(64, (3, 3), t=6, strides=2, n=4)
    self.block6 = InvertedResidualBlock(96, (3, 3), t=6, strides=1, n=3)
    self.block7 = InvertedResidualBlock(160, (3, 3), t=6, strides=2, n=3)  # ssd output
    self.block8 = InvertedResidualBlock(320, (3, 3), t=6, strides=1, n=1)
    self.block9 = ConvBlock(1280, (1, 1), strides=(1, 1))

  def call(self, inputs):
    output = self.block1(inputs)  # input 224*3
    output, _ = self.block2(output)  # input 112*32
    output, _ = self.block3(output)  # input 112*16
    output, _ = self.block4(output)  # input 56*24
    output, _ = self.block5(output)  # input 28*32
    output, _ = self.block6(output)  # input 14*64
    output, f1 = self.block7(output)  # input 14*96
    output, _ = self.block8(output)  # input 7*160
    output = self.block9(output)  # input 7*320
    return f1, output

  def init(self):
    origin_mobilenet_v2 = tf.keras.applications.MobileNetV2(weights='imagenet')
    len_mobilenet_v2 = 100
    weights = []
    for i in range(len_mobilenet_v2):
      print(i+1, (origin_mobilenet_v2.get_layer(index=i+1).name))
      weights.extend(origin_mobilenet_v2.get_layer(index=i+1).get_weights())

    self.set_weights(weights)


class ConvBlock(tf.keras.layers.Layer):
  """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
  """
  def __init__(self, filters, kernel, strides):
    super().__init__()

    self.conv = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides)
    self.bn = tf.keras.layers.BatchNormalization()
    self.relu6 = tf.keras.layers.ReLU(6.)

  def call(self, inputs):
    output = self.conv(inputs)
    output = self.bn(output)
    output = self.relu6(output)
    return output


class BottleNeck(tf.keras.layers.Layer):
  """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
  """
  def __init__(self, filters, kernel, t, s, r=False):

    self.t = t
    self.r = r

    self.pointwise_conv1 = None
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.relu6_1 = tf.keras.layers.ReLU(6.)

    self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.relu6_2 = tf.keras.layers.ReLU(6.)

    self.pointwise_conv2 = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')
    self.bn3 = tf.keras.layers.BatchNormalization()

  def build(self, input_shape):
    tchannel = input_shape[-1] * self.t
    self.pointwise_conv1 = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')

  def call(self, inputs):
    output = self.pointwise_conv1(inputs)
    output = self.bn1(output)
    output = self.relu6_1(output)
    feature_output

    output = self.depthwise_conv(output)
    output = self.bn2(output)
    output = self.relu6_2(output)

    output = self.pointwise_conv2(output)
    output = self.bn3(output)

    if self.r:
      output = tf.keras.layers.add([output, inputs])
    return output, feature_output


class InvertedResidualBlock(tf.keras.layers.Layer):
  """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
  """
  def __init__(self, filters, kernel, t, strides, n):
    super().__init__()

    self.bottleneck = BottleNeck(filters, kernel, t, strides)

    self.bottlenecks = []
    for i in range(1, n):
      self.bottlenecks.append(BottleNeck(filters, kernel, t, 1, True))

  def call(self, inputs):
    output, feature_output = self.bottleneck(inputs)

    for bn in self.bottlenecks:
        output, _ = bn(output)
    return output, feature_output
