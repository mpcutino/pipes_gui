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
        MainWindow.resize(870, 463)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.gBox_OneImageTest = QtWidgets.QGroupBox(self.centralwidget)
        self.gBox_OneImageTest.setEnabled(True)
        self.gBox_OneImageTest.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.gBox_OneImageTest.setTitle("")
        self.gBox_OneImageTest.setObjectName("gBox_OneImageTest")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.gBox_OneImageTest)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.lbl_image = QtWidgets.QLabel(self.gBox_OneImageTest)
        self.lbl_image.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_image.setObjectName("lbl_image")
        self.verticalLayout.addWidget(self.lbl_image)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_loadImage = QtWidgets.QPushButton(self.gBox_OneImageTest)
        self.btn_loadImage.setObjectName("btn_loadImage")
        self.horizontalLayout.addWidget(self.btn_loadImage)
        self.btn_SaveInputImg = QtWidgets.QPushButton(self.gBox_OneImageTest)
        self.btn_SaveInputImg.setObjectName("btn_SaveInputImg")
        self.horizontalLayout.addWidget(self.btn_SaveInputImg)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lbl_resultImg = QtWidgets.QLabel(self.gBox_OneImageTest)
        self.lbl_resultImg.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_resultImg.setObjectName("lbl_resultImg")
        self.verticalLayout_2.addWidget(self.lbl_resultImg)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.cbox_algorithm = QtWidgets.QComboBox(self.gBox_OneImageTest)
        self.cbox_algorithm.setEnabled(False)
        self.cbox_algorithm.setObjectName("cbox_algorithm")
        self.horizontalLayout_4.addWidget(self.cbox_algorithm)
        self.btn_SaveFilter = QtWidgets.QPushButton(self.gBox_OneImageTest)
        self.btn_SaveFilter.setObjectName("btn_SaveFilter")
        self.horizontalLayout_4.addWidget(self.btn_SaveFilter)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.lbl_pipesImg = QtWidgets.QLabel(self.gBox_OneImageTest)
        self.lbl_pipesImg.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.lbl_pipesImg.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_pipesImg.setObjectName("lbl_pipesImg")
        self.verticalLayout_3.addWidget(self.lbl_pipesImg)
        self.lbl_VIS_img = QtWidgets.QLabel(self.gBox_OneImageTest)
        self.lbl_VIS_img.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_VIS_img.setObjectName("lbl_VIS_img")
        self.verticalLayout_3.addWidget(self.lbl_VIS_img)
        self.btn_SaveCrop = QtWidgets.QPushButton(self.gBox_OneImageTest)
        self.btn_SaveCrop.setObjectName("btn_SaveCrop")
        self.verticalLayout_3.addWidget(self.btn_SaveCrop)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout_4.addWidget(self.gBox_OneImageTest)
        spacerItem = QtWidgets.QSpacerItem(20, 50, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout_4.addItem(spacerItem)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout_3.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.checkBox_UseFolder = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_UseFolder.setObjectName("checkBox_UseFolder")
        self.horizontalLayout_3.addWidget(self.checkBox_UseFolder)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.btn_loadInputPath = QtWidgets.QPushButton(self.centralwidget)
        self.btn_loadInputPath.setEnabled(False)
        self.btn_loadInputPath.setObjectName("btn_loadInputPath")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.btn_loadInputPath)
        self.lEdit_FilesLoadPath = QtWidgets.QLineEdit(self.centralwidget)
        self.lEdit_FilesLoadPath.setEnabled(False)
        self.lEdit_FilesLoadPath.setObjectName("lEdit_FilesLoadPath")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lEdit_FilesLoadPath)
        self.btn_LoadSaveFolder = QtWidgets.QPushButton(self.centralwidget)
        self.btn_LoadSaveFolder.setEnabled(False)
        self.btn_LoadSaveFolder.setObjectName("btn_LoadSaveFolder")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.btn_LoadSaveFolder)
        self.lEdit_FilesSavePath = QtWidgets.QLineEdit(self.centralwidget)
        self.lEdit_FilesSavePath.setEnabled(False)
        self.lEdit_FilesSavePath.setObjectName("lEdit_FilesSavePath")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lEdit_FilesSavePath)
        self.horizontalLayout_3.addLayout(self.formLayout_2)
        spacerItem2 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.btn_FilterSaveAll = QtWidgets.QPushButton(self.centralwidget)
        self.btn_FilterSaveAll.setEnabled(False)
        self.btn_FilterSaveAll.setObjectName("btn_FilterSaveAll")
        self.verticalLayout_5.addWidget(self.btn_FilterSaveAll)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_5.addWidget(self.progressBar)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout_4.setStretch(0, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 870, 22))
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
        self.btn_SaveInputImg.setText(_translate("MainWindow", "Save Input"))
        self.lbl_resultImg.setText(_translate("MainWindow", "result image"))
        self.btn_SaveFilter.setText(_translate("MainWindow", "Save Filter"))
        self.lbl_pipesImg.setText(_translate("MainWindow", "crop image"))
        self.lbl_VIS_img.setText(_translate("MainWindow", "VIS image"))
        self.btn_SaveCrop.setText(_translate("MainWindow", "Save Crop"))
        self.checkBox_UseFolder.setText(_translate("MainWindow", "Use folder"))
        self.btn_loadInputPath.setText(_translate("MainWindow", "Input Folder"))
        self.btn_LoadSaveFolder.setText(_translate("MainWindow", "Save Folder"))
        self.btn_FilterSaveAll.setText(_translate("MainWindow", "Filter and Save All"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
