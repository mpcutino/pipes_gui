import cv2

from app.image_processing.cut_methods.utils import get_image


def slide_window(img_path, filtered_contours, window_height=20):
    img = get_image(img_path)
    if len(filtered_contours):
        rect_contours = [cv2.boundingRect(cnt) for cnt in filtered_contours]
        rect_contours = sorted(rect_contours, key=lambda rect: rect[1])
        # print([r[2] for r in rect_contours])
        j = 0
        best_interval, best_sum_width = (-1, -1), 0
        while j < len(rect_contours):
            low_bound = rect_contours[j][1]
            up_bound = low_bound + rect_contours[j][-1]     # the y plus the height
            sum_width = 0
            for i in range(j, len(rect_contours)):
                y = rect_contours[i][1]
                # because it is sorted, there is no need to ask for the low bound condition, but...
                if low_bound <= y <= up_bound:
                    sum_width += rect_contours[i][2]
                else:
                    # the array is sorted by y
                    break
            j += 1
            if best_sum_width < sum_width:
                best_sum_width = sum_width
                best_interval = (low_bound, up_bound)
                print("sw:", best_sum_width)
        low_bound, up_bound = best_interval
        if low_bound > 0:
            low_bound = max(0, low_bound - window_height)
            up_bound = min(img.shape[0], up_bound + window_height)
            return img[low_bound:up_bound, :, :] if len(img.shape) == 3 else img[low_bound:up_bound, :]
    return img
