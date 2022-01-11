import cv2
import torch
import numpy as np

from app.image_processing.cut_methods.utils import to_three_shape, filter_contours, draw_colored_contours


def get_depth_pred(midas, img_path, transform, device="cpu"):
    img = cv2.imread(img_path)
    # img[:, :img.shape[1]//2, :] = 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_ = prediction.cpu().numpy()
    depth_ = (depth_ - depth_.min()) * (255 / (depth_.max() - depth_.min()))
    depth_ = depth_.astype(np.uint8)

    print(depth_.mean())
    ret, filtered_img = cv2.threshold(depth_, depth_.mean(), 255, cv2.THRESH_TOZERO)
    # contours, hierarchy = cv2.findContours(filtered_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    print(len(contours))

    gabor_th = to_three_shape(filtered_img)
    # filtered_contours = filter_contours(contours)
    draw_colored_contours(contours, None, gabor_th, use_rect=False)

    return contours, filtered_img


def load_midas_transform(model_type):
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return transform


def load_midas(model_type, device='cpu'):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    return midas


