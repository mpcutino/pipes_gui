import os

import cv2
from pathlib import Path


def otsu_pipes_portion(img_path):
    wp = Path(img_path)
    cd = os.getcwd()
    os.chdir(wp.parent)

    img = cv2.imread(wp.name)
    os.chdir(cd)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY +
                                 cv2.THRESH_OTSU)

    return thresh1
