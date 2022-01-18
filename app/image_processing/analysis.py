import numpy as np

from app.image_processing.cut_methods.utils import get_pipes_contour_lowup_bound, decide_broken_by_contours, \
    draw_img_surrounding_rect, draw_rects, get_image
from app.image_processing.cut_methods.standard_filter import gabor_pipes
from app.image_processing.cut_methods.portion_selection import sorted_x_slide_window


def broken_iterative_detection(img_p, thr_values, window_h, min_hit_percent=0.4):
    """
    A set of threshold values are used to compute the possible broken pipes.
    For each threshold, each pixel is updated by adding 1 if it corresponds to a possible broken envelope
    (defined by the contour rectangle), or 0 otherwise. The pixel matrix is initialized with zeros.
    The pixels considered as part of a broken glass envelope are those surpassing a given probability threshold.
    The pixel probability is simply computed as the division of the matrix value over the length of the threshold set.
    :param img_p:
    :param thr_values:
    :param window_h:
    :param min_hit_percent:
    :return:
    """
    img = get_image(img_p)
    hit_matrix = np.zeros((img.shape[0], img.shape[1]))
    processed = 0
    for thr in thr_values:
        filtered_contours, contour_img = \
            gabor_pipes(img_p, cond=lambda x, y, w, h: w * h > 50 and (h / w > 5 or w / h > 5), thr=thr)
        cut_img, vis_img = sorted_x_slide_window(img_p, filtered_contours, window_height=window_h)
        if cut_img is None:
            continue
        processed += 1
        l, u = get_pipes_contour_lowup_bound(filtered_contours, window_h, cut_img.shape[1])
        br_rects = decide_broken_by_contours(filtered_contours, l - window_h // 2, u + window_h // 2)
        # TODO!!!
        # cambiar el metodo de decision:
        #   posible mejora: usar un nuevo metodo que, en vez del area o la altura del rectangulo, utilice la menor
                    # distancia entre dos puntos horizontales del contorno con la misma y. Una forma de aproximar esto
                    # es, para cada contorno individual, encontrar el menor kernel tal que al erosionar el contorno con
                    # ese kernel, el contorno se divide. Hacerlo de forma iterativa: crear una imagen solo con el
                    # contorno que se analiza, inizializar un kernel de tamano 2, 2, y luego incrementarlo en la
                    # direccion vertical solamente, con tamano de paso 1. Aplicar findcontours y si hay mas de 1 o no
                    # hay ninguno, devolver el kernel. Si no, seguir incrementando.

        for (x, y, w, h) in br_rects:
            hit_matrix[y:y+h, x:x+w] += 1
    if processed > 0:
        hit_matrix /= processed
        hit_matrix[hit_matrix <= min_hit_percent] = 0
        # hit_matrix[hit_matrix >= min_hit_percent] = 1
        hit_matrix *= 255

    return hit_matrix.astype('uint8') if hit_matrix.sum() else None


def draw_cut_vis_image(cut_img, vis_img, filtered_contours, window_h):
    # CV evaluation of possible broken envelope
    l, u = get_pipes_contour_lowup_bound(filtered_contours, window_h, cut_img.shape[1])
    br_rects = decide_broken_by_contours(filtered_contours, l - window_h // 2, u + window_h // 2)
    if vis_img is not None:
        vis_img = draw_img_surrounding_rect(vis_img, (0, 0, 255) if len(br_rects) else (0, 255, 0))
    if cut_img is not None:
        cut_img = draw_rects(cut_img, br_rects, (0, 0, 255), y_translate=window_h - l)

    return cut_img, vis_img, br_rects
