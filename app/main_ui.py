import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage

from ui.design import Ui_MainWindow

from image_processing.cut_methods.otsu import otsu_pipes_portion


class MainApp(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        self.algorithms = ["depth", "otsu_threshold"]
        self.btn_loadImage.clicked.connect(self.load_image)

        for a in self.algorithms:
            self.cbox_algorithm.addItem(a)
        self.img_path = None

    def load_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Open image', filter="Image files (*.jpg *.png)")
        imagePath = fname[0]
        pixmap = QPixmap(imagePath)
        self.lbl_image.setPixmap(QPixmap(pixmap))
        self.lbl_image.setScaledContents(True)

        self.cbox_algorithm.setEnabled(True)
        self.img_path = imagePath

        self.set_result_image()

    def set_result_image(self):
        if self.img_path is not None:
            otsu_img = otsu_pipes_portion(self.img_path)

            format_ = QImage.Format_RGB888 if len(otsu_img.shape) == 3 else QImage.Format_Grayscale8
            height, width, channel = otsu_img.shape if len(otsu_img.shape) == 3 else otsu_img.shape[0], otsu_img.shape[1], 1

            bytesPerLine = channel * width
            result_qImg = QImage(otsu_img.data, width, height, bytesPerLine, format_).rgbSwapped()
            self.lbl_resultImg.setPixmap(QPixmap(result_qImg))
            self.lbl_resultImg.setScaledContents(True)


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(App.exec())
