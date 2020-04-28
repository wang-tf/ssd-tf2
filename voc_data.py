import tensorflow as tf
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import random
import glob

from box_utils import compute_target
from image_utils import random_patching, horizontal_flip
from functools import partial


class VOCDataset():
    """ Class for VOC Dataset

    Attributes:
        root_dir: dataset root dir (ex: ./data/VOCdevkit)
        year: dataset's year (2007 or 2012)
        num_examples: number of examples to be used
                      (in case one wants to overfit small data)
    """

    def __init__(self, root_dir, year, default_boxes,
                 new_size, num_examples=-1, augmentation=None):
        super(VOCDataset, self).__init__()
        self.idx_to_name = ['head']
        self.name_to_idx = dict([(v, k)
                                 for k, v in enumerate(self.idx_to_name)])

        self.data_dir = os.path.join(root_dir, 'VOC{}'.format(year))
        self.image_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.anno_dir = os.path.join(self.data_dir, 'Annotations')
        #self.ids = list(map(lambda x: x[:-4], os.listdir(self.image_dir)))
        self.default_boxes = default_boxes
        self.new_size = new_size

        if num_examples != -1:
            self.ids = self.ids[:num_examples]

        # self.train_ids = self.ids[:int(len(self.ids) * 0.75)]
        # self.val_ids = self.ids[int(len(self.ids) * 0.75):]
        train_name_path = os.path.join(self.data_dir, 'ImageSets/Main/train.txt')
        self.train_ids = []
        if os.path.exists(train_name_path):
            with open(train_name_path) as f:
                self.train_ids = f.read().strip().split('\n')
            
        val_name_path = os.path.join(self.data_dir, 'ImageSets/Main/val.txt')
        self.val_ids = []
        if os.path.exists(val_name_path):
            with open(val_name_path) as f:
                self.val_ids = f.read().strip().split('\n')

        test_name_path = os.path.join(self.data_dir, 'ImageSets/Main/test.txt')
        self.test_ids = []
        if os.path.exists(test_name_path):
            with open(test_name_path) as f:
                self.test_ids = f.read().strip().split('\n')
        self.ids = self.train_ids + self.val_ids + self.test_ids

        if augmentation == None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']

    def __len__(self):
        return len(self.ids)

    def _get_image(self, index):
        """ Method to read image from file

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (3, 300, 300)
        """
        filename = self.ids[index]
        img_path = os.path.join(self.image_dir, filename + '.jpg')
        img = Image.open(img_path)

        return img

    def _get_annotation(self, index):
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1

        Args:
            index: the index to get filename from self.ids

        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        filename = self.ids[index]
        anno_path = os.path.join(self.anno_dir, filename + '.xml')
        assert os.path.exists(anno_path), anno_path
        objects = ET.parse(anno_path).findall('object')
        boxes = []
        labels = []

        for obj in objects:
            name = obj.find('name').text.lower().strip()
            # filter useless obj
            # if name not in self.idx_to_name:
            #     continue

            bndbox = obj.find('bndbox')
            xmin = (float(bndbox.find('xmin').text) - 1)
            ymin = (float(bndbox.find('ymin').text) - 1)
            xmax = (float(bndbox.find('xmax').text) - 1)
            ymax = (float(bndbox.find('ymax').text) - 1)
            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(self.name_to_idx[name] + 1)

        boxes = np.array(boxes, dtype=np.float32)  # .reshape((-1, 4))
        labels = np.array(labels, dtype=np.int64)
        return boxes, labels

    def generate(self, subset=None):
        """ The __getitem__ method
            so that the object can be iterable
            the imaget was resized to (300, 300)
            then subtract by ImageNet's mean
            then convert to Tensor

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        if subset == 'train':
            indices = self.train_ids
        elif subset == 'val':
            indices = self.val_ids
        elif subset == 'test':
            indices = self.test_ids
        else:
            indices = self.ids
        for index in range(len(indices)):
            # img, orig_shape = self._get_image(index)
            filename = indices[index]
            img = self._get_image(index)
            w, h = img.size
            boxes, labels = self._get_annotation(index)  # the shape of boxes must not be (0, )

            # 存在没有标签的图片
            if boxes.shape[0] == 0:
                continue

            boxes = boxes / [w, h, w, h]
            try:
                assert boxes.shape[0] != 0, 'the shape of boxes must not be (0, )'
            except AssertionError as err:
                print(index, indices[index])
                raise
            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)

            augmentation_method = np.random.choice(self.augmentation)
            if augmentation_method == 'patch':
                img, boxes, labels = random_patching(img, boxes, labels)
            elif augmentation_method == 'flip':
                img, boxes, labels = horizontal_flip(img, boxes, labels)

            # resize image
            img = np.array(img.resize((self.new_size, self.new_size)), dtype=np.float32)
            # normalize
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)

            gt_confs, gt_locs = compute_target(self.default_boxes, boxes, labels)

            yield filename, img, gt_confs, gt_locs


def create_batch_generator(root_dir, year, default_boxes,
                           new_size, batch_size, num_batches,
                           mode,
                           augmentation=None):
    num_examples = batch_size * num_batches if num_batches > 0 else -1
    voc = VOCDataset(root_dir, year, default_boxes,
                     new_size, num_examples, augmentation)

    info = {
        'idx_to_name': voc.idx_to_name,
        'name_to_idx': voc.name_to_idx,
        'length': len(voc),
        'image_dir': voc.image_dir,
        'anno_dir': voc.anno_dir
    }

    if mode == 'train':
        train_gen = partial(voc.generate, subset='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32))
        val_gen = partial(voc.generate, subset='val')
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32))

        train_dataset = train_dataset.shuffle(40).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.take(num_batches), val_dataset.take(-1), info
    elif mode == 'test':
        test_gen = partial(voc.generate, subset='test')
        test_dataset = tf.data.Dataset.from_generator(
            test_gen, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = test_dataset.batch(batch_size)
        return dataset.take(num_batches), info
    else:
        dataset = tf.data.Dataset.from_generator(
            voc.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset.take(num_batches), info
