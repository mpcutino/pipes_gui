import os
import sys
import shutil
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore

from app.ui.design import Ui_MainWindow

from app.image_processing.analysis import draw_cut_vis_image, broken_iterative_detection
from app.image_processing.cut_methods.matching import gray_img_matching
from app.image_processing.cut_methods.standard_filter import pablo_otsu_pipes_portion, gabor_pipes
from app.image_processing.cut_methods.portion_selection import slide_window, sorted_x_slide_window

from save_popup_ui import SavePopupQWidget


class MainApp(QMainWindow, Ui_MainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        self.btn_loadImage.clicked.connect(self.load_image)
        self.lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_resultImg.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_pipesImg.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_VIS_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.btn_SaveInputImg.clicked.connect(self.save_input_img)
        self.btn_SaveFilter.clicked.connect(self.save_filter_img)
        self.btn_SaveCrop.clicked.connect(self.save_crop_img)

        self.algorithms = ["gabor_filter", "otsu_threshold", "simple_matching"]
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
        self.save_only_broken = False
        self.chBox_onlyBroken.stateChanged.connect(self.only_broken_change)

        self.progressBar.setValue(0)
        self.progressBar.setMaximum(100)

        self.spinBox_ThrValue.setValue(215)
        self.spinBox_ThrValue.valueChanged.connect(self.threshold_change)

        self.window_h = 20
        self.matching_img = "/home/mpcutino/Codes/pipes_gui/to_match.JPG"

    def only_broken_change(self, value):
        self.save_only_broken = value == QtCore.Qt.Checked

    def threshold_change(self, value):
        if self.algorithms[self.cbox_algorithm.currentIndex()] == "gabor_filter":
            self.change_result_image(self.cbox_algorithm.currentIndex())

    def filter_and_save_all(self):
        self.popup_w = SavePopupQWidget(self.cbox_algorithm.currentText())
        self.popup_w.exec_()
        if self.popup_w.result():
            self.progressBar.setValue(0)
            self.setEnabled(False)
            # the user says: Ok
            ext = self.popup_w.cBox_Ext.currentText()
            print(ext)
            tmp_img_path = self.img_path
            search_p = os.path.join(self.input_folder, "*.{0}".format(ext))
            search_results = glob(search_p)
            processed = 0
            for f in search_results:
                self.img_path = f
                qcrop = self.change_result_image(self.cbox_algorithm.currentIndex(), paint=False) \
                    if not self.save_only_broken else broken_iterative_detection(f, range(195, 225), self.window_h)
                processed += 1
                self.progressBar.setValue(int(100 * processed / len(search_results)))
                print(processed)

                if self.save_only_broken:
                    if qcrop is None:
                        continue
                    qcrop = self.get_QImg(qcrop)
                dest = os.path.join(self.save_folder, Path(f).name)
                if qcrop is None:
                    shutil.copy(f, dest)
                else:
                    qcrop.save(dest, format="png")
            self.img_path = tmp_img_path
            self.setEnabled(True)

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

            self.gbox_GaborParams.setEnabled(value == "gabor_filter")

    def change_result_image(self, al_index, paint=True):
        if self.img_path and 0 <= al_index < len(self.algorithms):
            algorithm = self.algorithms[al_index]
            filtered_contours, contour_img = None, None
            if algorithm == "otsu_threshold":
                filtered_contours, contour_img = pablo_otsu_pipes_portion(self.img_path)
            if algorithm == "gabor_filter":
                filtered_contours, contour_img = \
                    gabor_pipes(
                        self.img_path,
                        cond=lambda x, y, w, h: w * h > 50 and (h / w > 5 or w / h > 5),
                        thr=self.spinBox_ThrValue.value()
                    )
            if algorithm == "simple_matching":
                contour_img = gray_img_matching(self.img_path, self.matching_img)

            if contour_img is not None:
                if paint:
                    filter_qimg = self.get_QImg(contour_img) if algorithm != "simple_matching" else \
                        self.get_QImg(np.ones_like(contour_img[0], dtype='uint8') * 255)
                    self.lbl_resultImg.setPixmap(QPixmap(filter_qimg))
                    self.lbl_resultImg.setScaledContents(True)

                cut_img, vis_img = \
                    sorted_x_slide_window(self.img_path, filtered_contours, window_height=self.window_h) \
                        if algorithm != "simple_matching" else contour_img
                if vis_img is None and cut_img is not None:
                    vis_img = np.zeros_like(cut_img, dtype=np.uint8)
                if algorithm == "gabor_filter" and paint and cut_img is not None:
                    # CV evaluation of possible broken envelope
                    cut_img, vis_img, _ = draw_cut_vis_image(cut_img, vis_img, filtered_contours, self.window_h)
                if cut_img is not None:
                    detect_qimg = self.get_QImg(cut_img)
                    if paint:
                        self.lbl_pipesImg.setPixmap(QPixmap(detect_qimg))
                        self.lbl_pipesImg.setScaledContents(True)
                        # if vis_img is not None:
                        vis_qimg = self.get_QImg(vis_img)
                        self.lbl_VIS_img.setPixmap(QPixmap(vis_qimg))
                        self.lbl_VIS_img.setScaledContents(True)
                    return detect_qimg
        return None

    def use_folder_change(self, state):
        if state == QtCore.Qt.Checked:
            self.gBox_OneImageTest.setEnabled(False)
            self.btn_LoadSaveFolder.setEnabled(True)
            self.btn_loadInputPath.setEnabled(True)
            self.chBox_onlyBroken.setEnabled(self.cbox_algorithm.currentText() == "gabor_filter")
            if self.input_folder is not None and self.save_folder is not None:
                self.btn_FilterSaveAll.setEnabled(True)
        else:
            self.gBox_OneImageTest.setEnabled(True)
            self.btn_LoadSaveFolder.setEnabled(False)
            self.btn_loadInputPath.setEnabled(False)
            self.btn_FilterSaveAll.setEnabled(False)
            self.chBox_onlyBroken.setEnabled(False)

    @staticmethod
    def get_QImg(img):
        format_ = QImage.Format_RGB888 if len(img.shape) == 3 else QImage.Format_Grayscale8
        height, width, channel = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)

        bytesPerLine = channel * width
        result_qImg = QImage(img.data, width, height, bytesPerLine, format_)
        return result_qImg.rgbSwapped() if channel == 3 else result_qImg


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(App.exec())
