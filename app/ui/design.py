# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(765, 321)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.lbl_image = QtWidgets.QLabel(self.centralwidget)
        self.lbl_image.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_image.setObjectName("lbl_image")
        self.verticalLayout.addWidget(self.lbl_image)
        self.btn_loadImage = QtWidgets.QPushButton(self.centralwidget)
        self.btn_loadImage.setObjectName("btn_loadImage")
        self.verticalLayout.addWidget(self.btn_loadImage)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lbl_resultImg = QtWidgets.QLabel(self.centralwidget)
        self.lbl_resultImg.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_resultImg.setObjectName("lbl_resultImg")
        self.verticalLayout_2.addWidget(self.lbl_resultImg)
        self.cbox_algorithm = QtWidgets.QComboBox(self.centralwidget)
        self.cbox_algorithm.setEnabled(False)
        self.cbox_algorithm.setObjectName("cbox_algorithm")
        self.verticalLayout_2.addWidget(self.cbox_algorithm)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.lbl_pipesImg = QtWidgets.QLabel(self.centralwidget)
        self.lbl_pipesImg.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_pipesImg.setObjectName("lbl_pipesImg")
        self.verticalLayout_3.addWidget(self.lbl_pipesImg)
        self.lbl_explainFinalImg = QtWidgets.QLabel(self.centralwidget)
        self.lbl_explainFinalImg.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_explainFinalImg.setObjectName("lbl_explainFinalImg")
        self.verticalLayout_3.addWidget(self.lbl_explainFinalImg)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 765, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lbl_image.setText(_translate("MainWindow", "image goes here"))
        self.btn_loadImage.setText(_translate("MainWindow", "Load Image"))
        self.lbl_resultImg.setText(_translate("MainWindow", "result image"))
        self.lbl_pipesImg.setText(_translate("MainWindow", "(here goes the final image)"))
        self.lbl_explainFinalImg.setText(_translate("MainWindow", "Detection Result"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
