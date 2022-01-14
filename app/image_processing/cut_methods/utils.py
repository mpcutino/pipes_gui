import os
import random as rng
from pathlib import Path

import cv2
import numpy as np


def get_image(img_path):
    wp = Path(img_path)
    cd = os.getcwd()
    os.chdir(wp.parent)

    img = cv2.imread(wp.name)
    os.chdir(cd)
    return img


def filter_contours(contours, cond=None):
    filtered_contours = []
    if cond is None:
        cond = lambda x, y, w, h: w*h > 80 and (h/w > 5 or w/h > 5) and h < 25
    for i in range(len(contours)):
        cnt = contours[i]
        # x, y, w, h = cv2.boundingRect(cnt)
        (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)  # h, w is what works
        # print("----", w * h)
        # print((w, h, angle))
        if cond(cx, cy, w, h):
            filtered_contours.append(cnt)
    return filtered_contours


def to_three_shape(img):
    if len(img.shape) == 3:
        return img
    three_s = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    three_s[:, :, 0] = img
    three_s[:, :, 1] = img
    three_s[:, :, 2] = img
    return three_s


def draw_colored_contours(contours, hierarchy, img, use_rect=False):
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cnt = contours[i]
        if use_rect:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        else:
            cv2.drawContours(img, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
