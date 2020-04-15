#!/usr/bin/env python3
# coding:utf-8

import tensorflow as tf
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm
from absl import flags
from absl import app
from absl import logging

from anchor import generate_default_boxes
from box_utils import decode, compute_nms
from voc_data import create_batch_generator
from image_utils import ImageVisualizer
from losses import create_losses
from network import create_ssd
from PIL import Image


NUM_CLASSES = 2
BATCH_SIZE = 1


def get_args():
  FLAGS = flags.FLAGS
  flags.DEFINE_string('data_dir', '../dataset')
  flags.DEFINE_string('data_year', '2007')
  flags.DEFINE_string('arch', 'ssd800')
  flags.DEFINE_integer('num_examples', -1)
  flags.DEFINE_string('pretrained_type', 'specified')
  flags.DEFINE_string('checkpoint_dir', '')
  flags.DEFINE_string('checkpoint_path', '')
  flags.DEFINE_string('gpu_id', '0')

  return FLAGS


def predict(imgs, default_boxes):
  confs, locs = ssd(imgs)

  confs = tf.squeeze(confs, 0)
  locs = tf.squeeze(locs, 0)

  confs = tf.math.softmax(confs, axis=-1)
  classes = tf.math.argmax(confs, axis=-1)
  scores = tf.math.reduce_max(confs, axis=-1)

  boxes = decode(default_boxes, locs)

  out_boxes = []
  out_labels = []
  out_scores = []

  for c in range(1, NUM_CLASSES):
    cls_scores = confs[:, c]

    score_idx = cls_scores > 0.6
    # cls_boxes = tf.boolean_mask(boxes, score_idx)
    # cls_scores = tf.boolean_mask(cls_scores, score_idx)
    cls_boxes = boxes[score_idx]
    cls_scores = cls_scores[score_idx]

    nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 200)
    cls_boxes = tf.gather(cls_boxes, nms_idx)
    cls_scores = tf.gather(cls_scores, nms_idx)
    cls_labels = [c] * cls_boxes.shape[0]

    out_boxes.append(cls_boxes)
    out_labels.extend(cls_labels)
    out_scores.append(cls_scores)

  out_boxes = tf.concat(out_boxes, axis=0)
  out_scores = tf.concat(out_scores, axis=0)

  boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
  classes = np.array(out_labels)
  scores = out_scores.numpy()

  return boxes, classes, scores


def main{_}:
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
  with open('./config.yml') as f:
    cfg = yaml.load(f)

  try:
    config = cfg[FLAGS.arch.upper()]
  except AttributeError:
    raise ValueError('Unknown architecture: {}'.format(FLAGS.arch))

  default_boxes = generate_default_boxes(config)

  batch_generator, info = create_batch_generator(
        FLAGS.data_dir, FLAGS.data_year, default_boxes,
        config['image_size'],
        BATCH_SIZE, FLAGS.num_examples, mode='test')

  try:
    ssd = create_ssd(NUM_CLASSES, FLAGS.arch,
                         FLAGS.pretrained_type,
                         FLAGS.checkpoint_dir,
                         FLAGS.checkpoint_path)
  except Exception as e:
    print(e)
    print('The program is exiting...')
    sys.exit()

  os.makedirs('outputs/images', exist_ok=True)
  os.makedirs('outputs/detects', exist_ok=True)
  visualizer = ImageVisualizer(info['idx_to_name'], save_dir='outputs/images')

  for i, (filename, imgs, gt_confs, gt_locs) in enumerate(
    tqdm(batch_generator, total=info['length'],
             desc='Testing...', unit='images')):
    boxes, classes, scores = predict(imgs, default_boxes)
    filename = filename.numpy()[0].decode()
    original_image = Image.open(
            os.path.join(info['image_dir'], '{}.jpg'.format(filename)))
    boxes *= original_image.size * 2
    visualizer.save_image(
            original_image, boxes, classes, '{}.jpg'.format(filename))

    log_file = os.path.join('outputs/detects', '{}.txt')

    for cls, box, score in zip(classes, boxes, scores):
      cls_name = info['idx_to_name'][cls - 1]
      with open(log_file.format(cls_name), 'a') as f:
        f.write('{} {} {} {} {} {}\n'.format(filename, score, *[coord for coord in box]))


if __name__ == '__main__':
  FLAGS = get_args() 
  app.run(main)
