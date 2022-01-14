import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def gabor_pipes(img_path, cond=None):
    img = get_image(img_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret, blur = cv2.threshold(img_gray, 100, 255, cv2.THRESH_TOZERO)
    ret, blur = cv2.threshold(img_gray, 0, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)
    # blur = img_gray

    filtered_contours, gabor_th = get_gabor_contours(blur, cond=cond)

    return filtered_contours, gabor_th


def deginrad(degree):
    radiant = 2*np.pi/360 * degree
    return radiant


def get_gabor_contours(gray_img, thr=190, cond=None):
    # g_kernel = cv2.getGaborKernel((21, 21), 2.0, 0.9 * np.pi / 2, 10.0, 0.06, 0, ktype=cv2.CV_32F)
    gh_kernel = cv2.getGaborKernel((5, 5), 1, deginrad(90), 1, 0.1, 0, ktype=cv2.CV_32F)
    gh_kernel /= 1.0 * gh_kernel.sum()  # Brightness normalization
    # plt.imshow(gh_kernel)
    # plt.show()
    # plt.close()
    filteredh_img = cv2.filter2D(gray_img, cv2.CV_8UC3, gh_kernel)
    # ret, filteredh_img = cv2.threshold(filteredh_img, thr, 255, cv2.THRESH_BINARY)
    ret, filteredh_img = cv2.threshold(filteredh_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    gv_kernel = cv2.getGaborKernel((5, 5), 1, deginrad(0), 1, 0.1, 0, ktype=cv2.CV_32F)
    gv_kernel /= 1.0 * gv_kernel.sum()  # Brightness normalization
    filteredv_img = cv2.filter2D(gray_img, cv2.CV_8UC3, gv_kernel)

    ret, filteredv_img = cv2.threshold(filteredv_img, thr, 255, cv2.THRESH_BINARY)
    # ret, filteredv_img = cv2.threshold(filteredv_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    filtered_img = cv2.bitwise_and(filteredh_img, filteredh_img, mask=filteredv_img)
    # filtered_img = filteredh_img

    # erode and dilate the image, to prevent week connections
    # kernel = np.ones((4, 15), np.uint8)
    kernel = np.ones((2, 10), np.uint8)
    # only eroding
    filtered_img = cv2.erode(filtered_img, kernel, iterations=1)
    # filtered_img = cv2.erode(filtered_img, kernel, iterations=5)
    # filtered_img = cv2.dilate(filtered_img, np.ones((2, 2), np.uint8), iterations=1)
    # filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(filtered_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours) == 0:
        return [], filteredv_img
    hierarchy = hierarchy[0]    # for some reason, cv2 has a double list in python

    parents = {}
    for i in range(len(contours)):
        parents.setdefault(hierarchy[i][-1], 0)
        parents[hierarchy[i][-1]] += 1
    # print(parents)

    hierarchy_filtered_c = []
    for i in range(len(contours)):
        if i not in parents or parents[i] <= 1:
            hierarchy_filtered_c.append(contours[i])
    contours = hierarchy_filtered_c

    # area_filter_c = []
    # for cnt in contours:
    #     black_img = np.zeros(filtered_img.shape[:2], dtype=np.uint8)
    #     cv2.drawContours(black_img, [cnt], 0, 255, -1)
    #
    #     rect = cv2.minAreaRect(cnt)
    #     _, (w, h), _ = rect
    #
    #     masked = cv2.bitwise_and(filtered_img, filtered_img, mask=black_img)
    #     # plt.imshow(masked)
    #     # plt.show()
    #     # plt.close()
    #     c_area = (masked > 0).sum()
    #     print(c_area, w * h, w, h)
    #     if c_area >= 0.5*w*h and w*h > 0:
    #         print(c_area, w * h, w, h)
    #         area_filter_c.append(cnt)
    # print("elem area", len(area_filter_c))
    # contours = area_filter_c

    polygon_filter_c = []
    for cnt in contours:
        approx_p = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx_p) <= 7:
            polygon_filter_c.append(cnt)
    contours = polygon_filter_c

    gabor_th = to_three_shape(filtered_img)
    filtered_contours = filter_contours(contours, cond=cond)
    # filtered_contours = contours
    draw_colored_contours(filtered_contours, None, gabor_th, use_rect=False)

    return filtered_contours, gabor_th
    # return filtered_contours, filtered_img
