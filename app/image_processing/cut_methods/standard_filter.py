import cv2
import numpy as np

from app.image_processing.cut_methods.utils import get_image, to_three_shape, filter_contours, draw_colored_contours


def pablo_otsu_pipes_portion(img_path):

    img = get_image(img_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

    ret, thresh1 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    threeD_thresh = to_three_shape(thresh1)

    pablo_cond = lambda x, y, w, h: h * w > 1000 and h / w < (1 / 10)
    filtered_contours = filter_contours(contours, cond=pablo_cond)
    draw_colored_contours(filtered_contours, None, threeD_thresh, use_rect=False)

    return filtered_contours, threeD_thresh


def gabor_pipes(img_path):
    img = get_image(img_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret, blur = cv2.threshold(img_gray, 100, 255, cv2.THRESH_TOZERO)
    ret, blur = cv2.threshold(img_gray, 0, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)

    filtered_contours, gabor_th = get_gabor_contours(blur)

    return filtered_contours, gabor_th


def get_gabor_contours(gray_img, thr=180):
    g_kernel = cv2.getGaborKernel((21, 21), 2.0, 0.9 * np.pi / 2, 10.0, 0.06, 0, ktype=cv2.CV_32F)
    g_kernel /= 1.0 * g_kernel.sum()  # Brightness normalization
    filtered_img = cv2.filter2D(gray_img, cv2.CV_8UC3, g_kernel)

    ret, filtered_img = cv2.threshold(filtered_img, thr, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(filtered_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    gabor_th = to_three_shape(filtered_img)
    filtered_contours = filter_contours(contours)
    draw_colored_contours(filtered_contours, None, gabor_th, use_rect=False)

    return filtered_contours, gabor_th
