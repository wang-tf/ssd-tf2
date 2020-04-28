import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random
import numpy as np
import tensorflow as tf

from box_utils import compute_iou


class ImageVisualizer(object):
    """ Class for visualizing image

    Attributes:
        idx_to_name: list to convert integer to string label
        class_colors: colors for drawing boxes and labels
        save_dir: directory to store images
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = './'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def save_image(self, img, boxes, labels, name):
        """ Method to draw boxes and labels
            then save to dir

        Args:
            img: numpy array (width, height, 3)
            boxes: numpy array (num_boxes, 4)
            labels: numpy array (num_boxes)
            name: name of image to be saved
        """
        save_path = os.path.join(self.save_dir, name)

        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            left_top = (box[1], box[0])
            right_bottom = (box[3], box[2])
            print(box)
            box = list(map(int, box))
            cv2.rectangle(img, left_top, right_bottom, (0, 255, 0))
            cv2.putText(img, cls_name, left_top, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            #cv2.imshow('debug', img)
            #cv2.waitKey()

        cv2.imwrite(save_path, img)
        # raise


def generate_patch(boxes, threshold):
    """ Function to generate a random patch within the image
        If the patch overlaps any gt boxes at above the threshold,
        then the patch is picked, otherwise generate another patch

    Args:
        boxes: box tensor (num_boxes, 4)
        threshold: iou threshold to decide whether to choose the patch

    Returns:
        patch: the picked patch
        ious: an array to store IOUs of the patch and all gt boxes
    """
    while True:
        patch_w = random.uniform(0.1, 1)
        scale = random.uniform(0.5, 2)
        patch_h = patch_w * scale
        patch_xmin = random.uniform(0, 1 - patch_w)
        patch_ymin = random.uniform(0, 1 - patch_h)
        patch_xmax = patch_xmin + patch_w
        patch_ymax = patch_ymin + patch_h
        patch = np.array(
            [[patch_xmin, patch_ymin, patch_xmax, patch_ymax]],
            dtype=np.float32)
        patch = np.clip(patch, 0.0, 1.0)
        ious = compute_iou(tf.constant(patch), boxes)
        if tf.math.reduce_any(ious >= threshold):
            break

    return patch[0], ious[0]


def random_patching(img, boxes, labels):
    """ Function to apply random patching
        Firstly, a patch is randomly picked
        Then only gt boxes of which IOU with the patch is above a threshold
        and has center point lies within the patch will be selected

    Args:
        img: the original PIL Image
        boxes: gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)

    Returns:
        img: the cropped PIL Image
        boxes: selected gt boxes tensor (new_num_boxes, 4)
        labels: selected gt labels tensor (new_num_boxes,)
    """
    threshold = np.random.choice(np.linspace(0.1, 0.7, 4))

    patch, ious = generate_patch(boxes, threshold)

    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    keep_idx = (
        (ious > 0.3) &
        (box_centers[:, 0] > patch[0]) &
        (box_centers[:, 1] > patch[1]) &
        (box_centers[:, 0] < patch[2]) &
        (box_centers[:, 1] < patch[3])
    )

    if not tf.math.reduce_any(keep_idx):
        return img, boxes, labels

    img = img.crop(patch)

    boxes = boxes[keep_idx]
    patch_w = patch[2] - patch[0]
    patch_h = patch[3] - patch[1]
    boxes = tf.stack([
        (boxes[:, 0] - patch[0]) / patch_w,
        (boxes[:, 1] - patch[1]) / patch_h,
        (boxes[:, 2] - patch[0]) / patch_w,
        (boxes[:, 3] - patch[1]) / patch_h], axis=1)
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)

    labels = labels[keep_idx]

    return img, boxes, labels


def horizontal_flip(img, boxes, labels):
    """ Function to horizontally flip the image
        The gt boxes will be need to be modified accordingly

    Args:
        img: the original PIL Image
        boxes: gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)

    Returns:
        img: the horizontally flipped PIL Image
        boxes: horizontally flipped gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)
    """
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    boxes = tf.stack([
        1 - boxes[:, 2],
        boxes[:, 1],
        1 - boxes[:, 0],
        boxes[:, 3]], axis=1)

    return img, boxes, labels
