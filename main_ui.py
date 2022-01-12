import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage

import numpy as np
import matplotlib.pyplot as plt

from app.ui.design import Ui_MainWindow

from app.image_processing.cut_methods.depth_filter import load_midas, get_depth_pred, load_midas_transform
from app.image_processing.cut_methods.standard_filter import pablo_otsu_pipes_portion, gabor_pipes
from app.image_processing.cut_methods.portion_selection import slide_window


class MainApp(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        self.btn_loadImage.clicked.connect(self.load_image)
        self.lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_resultImg.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_pipesImg.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.algorithms = ["gabor_filter", "depth", "otsu_threshold"]
        self.cbox_algorithm.addItems(self.algorithms)
        self.cbox_algorithm.currentTextChanged.connect(self.filter_method_change)
        self.img_path = None

        # Model
        model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        self.depth_model = load_midas(model_type)
        self.depth_transform = load_midas_transform(model_type)

    def load_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Open image', filter="Image files (*.jpg *.png *.JPG)")
        imagePath = fname[0]
        pixmap = QPixmap(imagePath)
        self.lbl_image.setPixmap(QPixmap(pixmap))
        self.lbl_image.setScaledContents(True)

        self.cbox_algorithm.setEnabled(True)
        self.img_path = imagePath

        self.set_result_image()

    def set_result_image(self):
        if self.img_path is not None:
            filter_al = self.cbox_algorithm.currentIndex()
            self.change_result_image(filter_al)

    def filter_method_change(self, value):
        if value in self.algorithms:
            self.change_result_image(self.algorithms.index(value))

    def change_result_image(self, al_index):
        if self.img_path:
            filtered_contours, contour_img = None, None
            if al_index == 1:
                filtered_contours, contour_img = get_depth_pred(self.depth_model, self.img_path, self.depth_transform)
                # contour_img = (contour_img - contour_img.min())*(255/(contour_img.max() - contour_img.min()))
                # contour_img = contour_img.astype(np.uint8)
                plt.imshow(contour_img)
                plt.show()
                plt.close()
                print(contour_img.min(), contour_img.max())
                print(contour_img.shape)
            if al_index == 2:
                filtered_contours, contour_img = pablo_otsu_pipes_portion(self.img_path)
            if al_index == 0:
                filtered_contours, contour_img = \
                    gabor_pipes(self.img_path, cond=lambda x, y, w, h: w*h > 80 and (h/w > 5 or w/h > 5) and h < 100)
                # gabor_pipes(self.img_path, cond=lambda x, y, w, h: w*h > 80 and (h/w > 5 or w/h > 5) and h < 50)

            if contour_img is not None:
                filter_qimg = self.get_QImg(contour_img)
                self.lbl_resultImg.setPixmap(QPixmap(filter_qimg))
                self.lbl_resultImg.setScaledContents(True)

                cut_img = slide_window(self.img_path, filtered_contours)
                if cut_img is not None:
                    detect_qimg = self.get_QImg(cut_img)
                    self.lbl_pipesImg.setPixmap(QPixmap(detect_qimg))
                    self.lbl_pipesImg.setScaledContents(True)

    @staticmethod
    def get_QImg(img):
        format_ = QImage.Format_RGB888 if len(img.shape) == 3 else QImage.Format_Grayscale8
        height, width, channel = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)

        bytesPerLine = channel * width
        result_qImg = QImage(img.data, width, height, bytesPerLine, format_).rgbSwapped()

        return result_qImg


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(App.exec())
