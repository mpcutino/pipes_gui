import os
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore

import matplotlib.pyplot as plt

from app.ui.design import Ui_MainWindow

from app.image_processing.cut_methods.depth_filter import load_midas, get_depth_pred, load_midas_transform
from app.image_processing.cut_methods.standard_filter import pablo_otsu_pipes_portion, gabor_pipes
from app.image_processing.cut_methods.portion_selection import slide_window

from save_popup_ui import SavePopupQWidget


class MainApp(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        self.btn_loadImage.clicked.connect(self.load_image)
        self.lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_resultImg.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_pipesImg.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.btn_SaveInputImg.clicked.connect(self.save_input_img)
        self.btn_SaveFilter.clicked.connect(self.save_filter_img)
        self.btn_SaveCrop.clicked.connect(self.save_crop_img)

        self.algorithms = ["gabor_filter", "depth", "otsu_threshold"]
        self.cbox_algorithm.addItems(self.algorithms)
        self.cbox_algorithm.currentTextChanged.connect(self.filter_method_change)
        self.img_path = None
        self.input_folder = None
        self.save_folder = None

        self.checkBox_UseFolder.stateChanged.connect(self.use_folder_change)
        self.btn_loadInputPath.clicked.connect(self.load_input_folder)
        self.default_input_format = "JPG"
        self.btn_LoadSaveFolder.clicked.connect(self.load_save_folder)
        self.btn_FilterSaveAll.clicked.connect(self.filter_and_save_all)
        self.default_save_format = "png"
        self.popup_w = None

        # Model
        model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        self.depth_model = load_midas(model_type)
        self.depth_transform = load_midas_transform(model_type)

    def filter_and_save_all(self):
        self.popup_w = SavePopupQWidget(self.cbox_algorithm.currentText())
        # self.popup_w.setGeometry(QtCore.QRect(100, 100, 100, 120))
        # self.popup_w.show()
        self.popup_w.exec_()
        if self.popup_w.result():
            # the user says: Ok
            ext = self.popup_w.cBox_Ext.currentText()
            print(ext)

    def save_input_img(self):
        self.save_img(self.lbl_image.pixmap().toImage(), "Input_Image.png")

    def save_filter_img(self):
        self.save_img(self.lbl_resultImg.pixmap().toImage(), "Filtered_Image.png")

    def save_crop_img(self):
        self.save_img(self.lbl_pipesImg.pixmap().toImage(), "Cropped_Image.png")

    def save_img(self, qimg, img_name):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if len(folder):
            return qimg.save(os.path.join(folder, img_name))
        return False

    def load_input_folder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if len(folder):
            self.input_folder = folder
            if self.save_folder is not None:
                self.btn_FilterSaveAll.setEnabled(True)
            self.lEdit_FilesLoadPath.setText(folder)

    def load_save_folder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if len(folder):
            self.save_folder = folder
            if self.input_folder is not None:
                self.btn_FilterSaveAll.setEnabled(True)
            self.lEdit_FilesSavePath.setText(folder)

    def load_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Open image', filter="Image files (*.jpg *.png *.JPG)")
        imagePath = fname[0]
        if len(imagePath):
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

    def use_folder_change(self, state):
        if state == QtCore.Qt.Checked:
            self.gBox_OneImageTest.setEnabled(False)
            self.btn_LoadSaveFolder.setEnabled(True)
            self.btn_loadInputPath.setEnabled(True)
            if self.input_folder is not None and self.save_folder is not None:
                self.btn_FilterSaveAll.setEnabled(True)
        else:
            self.gBox_OneImageTest.setEnabled(True)
            self.btn_LoadSaveFolder.setEnabled(False)
            self.btn_loadInputPath.setEnabled(False)
            self.btn_FilterSaveAll.setEnabled(False)

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
