import os
import random as rng
from pathlib import Path

import cv2
import numpy as np


def get_image(img_path):
    wp = Path(img_path)
    cd = os.getcwd()
    os.chdir(wp.parent)

    if not os.path.exists(wp.name):
        return None
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
        if cond(cx, cy, w + 1, h + 1):
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


def get_pipes_contour_lowup_bound(filtered_contours, window_height, img_width):
    rect_contours = [cv2.boundingRect(cnt) for cnt in filtered_contours]
    rect_contours = sorted(rect_contours, key=lambda rect: rect[0])
    # print([r[0] for r in rect_contours])

    best_interval, best_sum_width = (-1, -1), 0
    for ind_r, r in enumerate(rect_contours):
        low_bound = r[1] - window_height // 2
        up_bound = r[1] + r[-1] + window_height // 2  # the y plus the height

        sum_width = r[2]
        next_lower_x = r[0] + r[2]
        for i in range(len(rect_contours)):
            if ind_r == i:
                continue
            y = rect_contours[i][1]
            x = rect_contours[i][0]
            # the new contour must be between the height of the analysed contour, and should be completely after
            #    the analysed contour.
            if low_bound <= y <= up_bound and x >= next_lower_x:
                sum_width += rect_contours[i][2]
                next_lower_x = x + rect_contours[i][2]
        # print(sum_width)
        if best_sum_width < sum_width <= img_width:
            best_sum_width = sum_width
            best_interval = (low_bound + window_height // 2, up_bound - window_height // 2)
            # best_interval = low_bound, up_bound
            # print("sw:", best_sum_width)
    return best_interval


def decide_broken_by_contours(filtered_contours, low_bound, up_bound, width_min=125, height_percent_thr=0.6):
    rect_contours = [(cv2.boundingRect(cnt), cv2.contourArea(cnt)) for cnt in filtered_contours]
    # reduce rect y value to cope with the cropped image
    rect_contours = [((x, y, w, h), area) for (x, y, w, h), area in rect_contours
                     if low_bound <= y <= up_bound and w >= width_min]
    if len(rect_contours) == 0:
        return []

    max_rect_cnt = max(rect_contours, key=lambda cnt: cnt[-1])
    possible_broken_rects = [r[0] for r in rect_contours if r[-1]/max_rect_cnt[-1] <= height_percent_thr]
    return possible_broken_rects


def draw_img_surrounding_rect(img, color, xy=(0, 0), wh=None, thickness=3):
    img = to_three_shape(img)
    if wh is None:
        wh = (img.shape[1], img.shape[0])
    nx, ny = xy[0] + wh[0], xy[1] + wh[1]
    cv2.rectangle(img, xy, (nx, ny), color=color, thickness=thickness)
    return img


def draw_rects(img, rects, color, thickness=1, y_translate=0):
    for x, y, w, h in rects:
        img = draw_img_surrounding_rect(img, color, (x, y + y_translate), (w, h), thickness=thickness)
    return img
