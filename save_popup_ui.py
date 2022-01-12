from PyQt5 import QtWidgets
from PyQt5 import QtCore

from app.ui.popup import Ui_Form


class SavePopupQWidget(QtWidgets.QDialog, Ui_Form):
    
    def __init__(self, alg):
        QtWidgets.QDialog.__init__(self)
        self.setupUi(self)

        self.lbl_Header.setText("Images will be saved using {0} algorithm.".format(alg))
        self.lbl_Ext.setText("Check the input extension!!")

        self.cBox_Ext.addItems(["JPG", "jpg", "png", "all"])

        self.btn_Cancel.clicked.connect(self.cancel)
        self.btn_Cancel.setText("Cancel")
        self.btn_Ok.clicked.connect(self.ok)
        self.btn_Ok.setText("Ok")

        # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, False)
        # self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        # self.setWindowFlag(QtCore.Qt.MSWindowsFixedSizeDialogHint, False)

    def cancel(self):
        self.reject()
        self.close()

    def ok(self):
        self.accept()
        self.close()
