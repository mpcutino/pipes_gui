# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'popup.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(204, 104)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.lbl_Header = QtWidgets.QLabel(Form)
        self.lbl_Header.setObjectName("lbl_Header")
        self.verticalLayout.addWidget(self.lbl_Header)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.lbl_Ext = QtWidgets.QLabel(Form)
        self.lbl_Ext.setObjectName("lbl_Ext")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lbl_Ext)
        self.cBox_Ext = QtWidgets.QComboBox(Form)
        self.cBox_Ext.setObjectName("cBox_Ext")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.cBox_Ext)
        self.btn_Ok = QtWidgets.QPushButton(Form)
        self.btn_Ok.setObjectName("btn_Ok")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.btn_Ok)
        self.btn_Cancel = QtWidgets.QPushButton(Form)
        self.btn_Cancel.setObjectName("btn_Cancel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.btn_Cancel)
        self.verticalLayout.addLayout(self.formLayout)
        self.verticalLayout.setStretch(1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.lbl_Header.setText(_translate("Form", "TextLabel"))
        self.lbl_Ext.setText(_translate("Form", "TextLabel"))
        self.btn_Ok.setText(_translate("Form", "PushButton"))
        self.btn_Cancel.setText(_translate("Form", "PushButton"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
