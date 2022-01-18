import cv2
import numpy as np
import matplotlib.pyplot as plt

from app.image_processing.cut_methods.utils import get_image


def gray_img_matching(img_path, matching_path):
    img = get_image(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    w = get_image(matching_path)
    w = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)

    img = img/255
    w = w/255

    min_d, mi, mj = None, -1, -1
    for i in range(img.shape[0]):
        y_range = img.shape[1] - w.shape[1] + 1
        for j in range(y_range):
            s = img[i:i+w.shape[0], j:j+w.shape[1]]
            if s.shape == w.shape:
                d = np.sqrt((s - w)**2).sum()
                if min_d is None or min_d > d:
                    min_d, mi, mj = d, i, j
    print("best match: dist-{0}, coords-{1}".format(min_d, (mi, mj)))

    vis_p = img_path.replace("_IR.JPG", "_VIS.jpg")
    vis_img = get_image(vis_p)

    vis_low_bound = int(mi * vis_img.shape[0] / img.shape[0]) - 5
    vis_up_bound = int((mi + w.shape[0]) * vis_img.shape[0] / img.shape[0]) + 5

    vis_img = vis_img[vis_low_bound:vis_up_bound, :, :] \
        if len(vis_img.shape) == 3 else vis_img[vis_low_bound:vis_up_bound, :]
    print(vis_img.shape)

    # img = img[mi:mi+w.shape[0], mj:mj+w.shape[1]] if mi >= 0 else img
    img = img[mi:mi+w.shape[0], :] if mi >= 0 else img

    return (img*255).astype('uint8'), vis_img


def sift_matching(img_path, matching_path):
    img = get_image(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    w = get_image(matching_path)
    w = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)

    # sift
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(w, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img, keypoints_1, w, keypoints_2, matches[:50], w, flags=2)
    plt.imshow(img3)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # im_ = gray_img_matching("/home/mpcutino/Downloads/P1_Ex1/P1_Ex1/VM_20210708092509_IR.JPG",
    #                         "/home/mpcutino/Codes/pipes_gui/to_match.JPG")
    # im_ = gray_img_matching("/home/mpcutino/Downloads/P1_Ex1/P1_Ex1/VM_20210708092514_IR.JPG",
    #                         "/home/mpcutino/Codes/pipes_gui/to_match.JPG")
    # print(im_.dtype)

    # sift_matching("/home/mpcutino/Downloads/P1_Ex1/P1_Ex1/VM_20210708092509_IR.JPG",
    #               "/home/mpcutino/Codes/pipes_gui/to_match.JPG")
    sift_matching("/home/mpcutino/Downloads/P1_Ex1/P1_Ex1/VM_20210708102543_IR.JPG",
                  "/home/mpcutino/Codes/pipes_gui/to_match.JPG")
    sift_matching("/home/mpcutino/Downloads/P1_Ex1/P1_Ex1/VM_20210708102543_IR.JPG",
                  "/home/mpcutino/Downloads/Test/VM_20210708092509_IR.JPG")
    sift_matching("/home/mpcutino/Downloads/P1_Ex1/P1_Ex1/VM_20210708102543_IR.JPG",
                  "/home/mpcutino/Downloads/P1_Ex1/P1_Ex1/VM_20210708092509_IR.JPG")
