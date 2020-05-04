#!/usr/bin/env python3
# coding:utf-8


import os
import sys
import time
import yaml
from absl import flags
from absl import app
import tensorflow as tf

from voc_data import create_batch_generator
from anchor import generate_default_boxes
# from ssd import create_ssd
from ssd_mobilenetv2 import create_ssd
from losses import create_losses

# need include background
NUM_CLASSES = 2


def get_args():
  FLAGS = flags.FLAGS
  flags.DEFINE_string('data_dir', None, 'input voc data dir')
  flags.DEFINE_string('data_year', '2007', 'VOC data year')
  flags.DEFINE_string('arch', 'ssd800', 'network format')
  flags.DEFINE_integer('batch_size', 6, 'batch size')
  flags.DEFINE_integer('num_batches', -1, 'if -1, use all data')
  flags.DEFINE_integer('neg_ratio', 3, 'negative positive example ratio')
  flags.DEFINE_float('initial_lr', 0.0006, 'initial learning rate')
  flags.DEFINE_float('momentum', 0.9, '')
  flags.DEFINE_float('weight_decay', 5e-4, '')
  flags.DEFINE_integer('num_epochs', 400, 'epoch number')
  flags.DEFINE_string('checkpoint_dir', './checkpoints', 'checkpoint save dir')
  flags.DEFINE_string('pretrained_type', 'base', '')
  flags.DEFINE_string('gpus', '0', 'gpus using')

  flags.mark_flag_as_required('data_dir')
  return FLAGS


#@tf.function
def train_step(imgs, gt_confs, gt_locs, ssd, criterion, optimizer):
  with tf.GradientTape() as tape:
    confs, locs = ssd(imgs)

    conf_loss, loc_loss = criterion(confs, locs, gt_confs, gt_locs)

    loss = conf_loss + loc_loss
    l2_loss = [tf.nn.l2_loss(t) for t in ssd.trainable_variables]
    l2_loss = FLAGS.weight_decay * tf.math.reduce_sum(l2_loss)
    loss += l2_loss

  gradients = tape.gradient(loss, ssd.trainable_variables)
  optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))

  return loss, conf_loss, loc_loss, l2_loss


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

  os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)

  with open('./config.yml') as f:
    cfg = yaml.load(f)

  try:
    config = cfg[FLAGS.arch.upper()]
  except AttributeError:
    raise ValueError('Unknown architecture: {}'.format(FLAGS.arch))

  default_boxes = generate_default_boxes(config)

  batch_generator, val_generator, info = create_batch_generator(
    FLAGS.data_dir,
    FLAGS.data_year,
    default_boxes,
    config['image_size'],
    FLAGS.batch_size,
    FLAGS.num_batches,
    mode='train',
    augmentation=[
      'flip'
    ])  # the patching algorithm is currently causing bottleneck sometimes

  try:
    ssd = create_ssd(NUM_CLASSES,
             FLAGS.arch,
             FLAGS.pretrained_type,
             checkpoint_dir=FLAGS.checkpoint_dir)
  except Exception as e:
    print(e)
    print('Can not create ssd. The program is exiting...')
    sys.exit()

  criterion = create_losses(FLAGS.neg_ratio, NUM_CLASSES)

  steps_per_epoch = info['length'] // FLAGS.batch_size

  lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[
    int(steps_per_epoch * FLAGS.num_epochs * 2 / 3),
    int(steps_per_epoch * FLAGS.num_epochs * 5 / 6)
  ],
                   values=[
                     FLAGS.initial_lr, FLAGS.initial_lr * 0.1,
                     FLAGS.initial_lr * 0.01
                   ])

  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_fn,
                    momentum=FLAGS.momentum)

  train_log_dir = 'logs/train'
  val_log_dir = 'logs/val'
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  val_summary_writer = tf.summary.create_file_writer(val_log_dir)

  for epoch in range(FLAGS.num_epochs):
    avg_loss = 0.0
    avg_conf_loss = 0.0
    avg_loc_loss = 0.0
    start = time.time()
    for i, (_, imgs, gt_confs, gt_locs) in enumerate(batch_generator):
      loss, conf_loss, loc_loss, l2_loss = train_step(
        imgs, gt_confs, gt_locs, ssd, criterion, optimizer)
      avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
      avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
      avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
      # if (i) % 50 == 0:
    print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'
          .format(epoch + 1, i + 1,
              time.time() - start, avg_loss, avg_conf_loss,
              avg_loc_loss))

    avg_val_loss = 0.0
    avg_val_conf_loss = 0.0
    avg_val_loc_loss = 0.0
    for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
      val_confs, val_locs = ssd(imgs)

      val_conf_loss, val_loc_loss = criterion(val_confs, val_locs, gt_confs, gt_locs)
      val_loss = val_conf_loss + val_loc_loss
      avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
      avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
      avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

    with train_summary_writer.as_default():
      tf.summary.scalar('loss', avg_loss, step=epoch)
      tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
      tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

    with val_summary_writer.as_default():
      tf.summary.scalar('loss', avg_val_loss, step=epoch)
      tf.summary.scalar('conf_loss', avg_val_conf_loss, step=epoch)
      tf.summary.scalar('loc_loss', avg_val_loc_loss, step=epoch)

    if (epoch + 1) % 10 == 0:
      h5_file_path = os.path.join(FLAGS.checkpoint_dir, f'ssd_epoch_{epoch+1}.h5')
      ssd.save_weights(h5_file_path)


if __name__ == '__main__':
  FLAGS = get_args()
  app.run(main)

