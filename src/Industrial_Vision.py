#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Copyright 2023 Alexis CORBILLET

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import time

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from pyueye import ueye
import cv2
from PIL import Image
import webcolors

from serial import Serial
import serial.tools.list_ports


class IAFormes:
    """
    IAFormes class implements a shape recognition system using a multi-layer perceptron (MLP) classifier.

    Parameters:
        Test (bool): If True, initializes the classifier with a test dataset. Default is False.

    Attributes:
        __test (bool): Flag indicating whether the test mode is enabled.
        __mlp2 (MLPClassifier): Multi-layer perceptron classifier object.
        x_test (ndarray): Test input data.
        y_test (ndarray): Test target labels.
        xt (ndarray): Training input data.
        yt (ndarray): Training target labels.

    """

    def __init__(self, Test=False):
        """
        Initializes the IAFormes object.

        Parameters:
            Test (bool): If True, initializes the classifier with a test dataset. Default is False.
        """

        self.__test = Test

        if Test:
            # Reading of the test DataBase
            dfAt = pd.read_csv(r'BaseATEST.csv')
            dfCt = pd.read_csv(r'BaseCercleTEST.csv')
            dfTt = pd.read_csv(r'BaseTriTEST.csv')
            dfPt = pd.read_csv(r'BasePentTEST.csv')
            dfEt = pd.read_csv(r'BaseEtTEST.csv')

            xt1 = [dfAt[str(i)].values for i in range(500)]
            xt2 = [dfCt[str(i)].values for i in range(500)]
            xt3 = [dfTt[str(i)].values for i in range(500)]
            xt4 = [dfPt[str(i)].values for i in range(500)]
            xt5 = [dfEt[str(i)].values for i in range(500)]

            xt = xt1+xt2+xt3+xt4+xt5+[[0]*3264]*500
            xt = np.array(xt)

            yt = [0]*(500)+[1]*(500)+[2]*(500)+[3]*(500)+[4]*(500)+[-1]*500
            yt = np.array(yt)
            x_test = xt
            y_test = yt
            self.xt = xt
            self.yt = yt

        x = np.zeros((6, 3264))
        y = yt = [0]+[1]+[2]+[3]+[4]+[-1]
        y = np.array(y)

        self.__mlp2 = MLPClassifier(hidden_layer_sizes=([256, 128]), activation='logistic', alpha=1e-4, solver='sgd',
                                    tol=5e-3, random_state=0, verbose=False, max_iter=1, warm_start=False, learning_rate_init=0.005)

        self.__mlp2.fit(x, y)

        A = np.load("Model_parameters/Poids-1.npy", allow_pickle=True)
        A = [A[0], A[1], A[2]]
        B = np.load("Model_parameters/Biais-1.npy", allow_pickle=True)
        B = [B[0], B[1], B[2]]

        self.__mlp2.coefs_ = A
        self.__mlp2.intercepts_ = B

        if Test:
            """
            Prints the accuracy score of the MLP classifier on the test dataset.
            """

            print("Test set score: %f" % self.__mlp2.score(x_test, y_test))

    def montreimage(self, k, set):
        """
        Displays the image at index k from the given set.

        Parameters:
            k (int): Index of the image to display.
            set (ndarray): Set of images.

        """

        X = np.reshape(set[k], (51, 64))
        plt.imshow(X, cmap='Greys_r')
        plt.show()

    def matconf(self):
        """
        Displays the confusion matrix based on the predicted labels and true labels of the test dataset.

        Note:
            This method should only be called when the test mode is enabled.
        """

        if self.__test:
            y_pred = self.__mlp2.predict(self.xt)
            cm = confusion_matrix(self.yt, y_pred)
            cm_display = ConfusionMatrixDisplay(cm).plot()
            plt.show()

        else:
            print("No test dataset")

    def Proba(self, xtest):
        """
        Predicts the probabilities of each class label for the given input data.

        Parameters:
            xtest (ndarray): Input data to predict probabilities for.

        Returns:
            ndarray: Array of predicted probabilities for each class label.

        """

        return self.__mlp2.predict_proba(xtest)

    def Proba2(self, xtest):
        """
        Predicts the class labels for the given input data.

        Parameters:
            xtest (ndarray): Input data to predict labels for.

        Returns:
            ndarray: Array of predicted labels.

        """

        return self.__mlp2.predict(xtest)


class Couleurim:
    """
    Couleurim class represents an image color analysis tool.

    Parameters:
        Foldername (str): The path to the image file. Default is 'images/o1.bmp'.
        livemode (bool): If True, uses a live frame instead of reading an image file. Default is False.
        Frame (PIL.Image.Image): The live frame to analyze. Only used if livemode is True.

    Attributes:
        r (ndarray): Red channel values of the image.
        g (ndarray): Green channel values of the image.
        b (ndarray): Blue channel values of the image.
        argr (tuple): Indices of the maximum value in the red channel.
        argg (tuple): Indices of the maximum value in the green channel.
        argb (tuple): Indices of the maximum value in the blue channel.
        nc (list): Maximum color values for each channel.
    """

    def __init__(self, Foldername='images/o1.bmp', livemode=False, Frame=None):
        """
        Initializes the Couleurim object.

        Parameters:
            Foldername (str): The path to the image file. Default is 'images/o1.bmp'.
            livemode (bool): If True, uses a live frame instead of reading an image file. Default is False.
            Frame (PIL.Image.Image): The live frame to analyze. Only used if livemode is True.
        """

        if livemode == False:
            img = Image.open(Foldername)
        else:
            img = Frame

        r, g, b = img.split()

        self.r = np.array(r)
        self.g = np.array(g)
        self.b = np.array(b)

        (n, p) = self.r.shape

        r = self.r.max()
        self.argr = np.unravel_index(self.r.argmax(), (n, p))

        g = self.g.max()
        self.argg = np.unravel_index(self.g.argmax(), (n, p))

        b = self.b.max()
        self.argb = np.unravel_index(self.b.argmax(), (n, p))

        self.nc = [r, g, b]

        ma = self.nc.index(max(self.nc))

        if ma == 0:
            self.argb = self.argr
            self.argg = self.argr

        if ma == 1:
            self.argb = self.argg
            self.argr = self.argg

        if ma == 2:
            self.argr = self.argb
            self.argg = self.argb

        self.nc = [self.r[self.argr], self.g[self.argg], self.b[self.argb]]

    def affichage(self):
        """
        Displays the image with highlighted color channels.

        Note:
            The red channel is shown in the top-left subplot.
            The green channel is shown in the bottom-left subplot.
            The blue channel is shown in the top-right subplot.
            The closest color name to the analyzed color is shown in the bottom-right subplot.
        """

        r = self.r
        b = self.b
        g = self.g
        nc = self.nc
        (n, p) = r.shape
        f = 5

        for k in range(p):
            for i in range(-f, f):
                if (self.argr[0]+i) > 0 and (self.argr[0]+i) < n:
                    r[self.argr[0]+i, k] = 255
                if (self.argg[0]+i) > 0 and (self.argg[0]+i) < n:
                    g[self.argg[0]+i, k] = 255
                if (self.argb[0]+i) > 0 and (self.argb[0]+i) < n:
                    b[self.argb[0]+i, k] = 255

        for k in range(n):
            for i in range(-f, f):
                if (self.argr[1]+i) > 0 and (self.argr[1]+i) < p:
                    r[k, self.argr[1]+i] = 255
                if (self.argg[1]+i) > 0 and (self.argg[1]+i) < p:
                    g[k, self.argg[1]+i] = 255
                if (self.argb[1]+i) > 0 and (self.argb[1]+i) < p:
                    b[k, self.argb[1]+i] = 255

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(r, cmap='Reds_r')
        ax[0, 0].set_title("Red nuances")
        ax[0, 0].axis('off')
        ax[1, 0].imshow(g, cmap='Greens_r')
        ax[1, 0].set_title("Green nuances")
        ax[1, 0].axis('off')
        ax[0, 1].imshow(b, cmap='Blues_r')
        ax[0, 1].set_title("Blue nuances")
        ax[0, 1].axis('off')
        ax[1, 1].imshow([[nc]])
        ax[1, 1].set_title(self.couleur_proche(nc))
        plt.show()

    def couleur_proche(self, requested_colour=None):
        """
        Finds the closest color name to the analyzed color.

        Parameters:
            requested_colour (list): RGB values of the color to find the closest name for. Default is None,
                                     which uses the analyzed color stored in nc attribute.

        Returns:
            str: The name of the closest color.
        """

        if requested_colour is None:
            requested_colour = self.nc

        min_colours = {}

        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name

        return min_colours[min(min_colours.keys())]


class IAcouleurs:
    """
    IAcouleurs class represents an image color and shape classification tool.

    Parameters:
        livemode (bool): If True, enables live mode for analyzing frames. Default is True.

    Attributes:
        IA (IAFormes): Instance of the IAFormes class for shape classification.
        livemode (bool): Flag indicating whether the live mode is enabled or not.
    """

    def __init__(self, livemode=True):
        """
        Initializes the IAcouleurs object.

        Parameters:
            livemode (bool): If True, enables live mode for analyzing frames. Default is True.
        """

        self.IA = IAFormes()
        self.__livemode = livemode

    def classification(self, s=50, Fichier='images/e1.bmp', Frame=None):
        """
        Performs color and shape classification on an image or frame.

        Parameters:
            s (int): Threshold value for image binarization. Default is 50.
            Fichier (str): Path to the image file. Default is 'images/e1.bmp'.
            Frame (ndarray): The live frame to classify. Only used if livemode is True.

        Returns:
            list: A list containing the classification results. The list contains the following elements:
                - The shape classification result as a tuple (shape_name, probability).
                - The closest color name to the analyzed color.
                - The RGB values of the analyzed color.
        """

        if self.__livemode == False:
            self.c = Couleurim(Foldername=Fichier)
            img = Image.open(Fichier)
        else:
            Frame = Image.fromarray(Frame, mode="RGB")
            self.c = Couleurim(livemode=True, Frame=Frame)
            img = Frame

        img = img.convert('L')
        img = np.array(img)

        Xt = self.aquisi(img, s)

        P = self.IA.Proba(Xt)
        self.L = {"Empty": "%0.2f" % P[0][0], "Square": "%0.2f" % P[0][1], "Cercle": "%0.2f" % P[0][2],
                  "Triangle": "%0.2f" % P[0][3], "Pentagone": "%0.2f" % P[0][4], "Star": "%0.2f" % P[0][5]}
        
        pred = self.IA.Proba2(Xt)

        return [list(self.L.items())[pred[0]+1], self.c.couleur_proche(), self.c.nc]

    def classi2(self, s=50, Fichier='images/e1.bmp', Frame=None):
        """
        Performs shape classification on an image or frame.

        Parameters:
            s (int): Threshold value for image binarization. Default is 50.
            Fichier (str): Path to the image file. Default is 'images/e1.bmp'.
            Frame (ndarray): The live frame to classify. Only used if livemode is True.

        Returns:
            int: The index of the predicted shape class.
        """

        if self.__livemode == False:
            self.c = Couleurim(Foldername=Fichier)
            img = Image.open(Fichier)
        else:
            Frame = Image.fromarray(Frame, mode="RGB")
            img = Frame

        img = img.convert('L')
        img = np.array(img)
        Xt = self.aquisi(img, s)
        pred = self.IA.Proba2(Xt)

        return pred

    def resize(self, image, l, m):
        """
        Resizes an image to the specified dimensions.

        Parameters:
            image (ndarray): The image to resize.
            l (int): The width of the resized image.
            m (int): The height of the resized image.

        Returns:
            ndarray: The resized image.
        """

        image = Image.fromarray(np.uint8((image)))
        image.thumbnail((l, m))

        return np.array(image)

    def seuil(self, imNB, s=120):
        """
        Applies a threshold to an image.

        Pixels with intensity values below the threshold are set to 0, and pixels with intensity values
        above or equal to the threshold are kept as is.

        Parameters:
            imNB (ndarray): The grayscale image to apply the threshold to.
            s (int): Threshold value. Default is 120.

        Returns:
            ndarray: The thresholded image.
        """

        return np.where(imNB < s, 0, imNB)

    def aquisi(self, im, s):
        """
        Preprocesses an image for shape classification.

        This method applies thresholding and resizing to the input image.

        Parameters:
            im (ndarray): The grayscale image to preprocess.
            s (int): Threshold value for image binarization.

        Returns:
            ndarray: Preprocessed image data as a 1D array.
        """

        (n, p) = im.shape
        im = self.seuil(im, s)
        im = self.resize(im, 64, 64)
        T = [im]
        T = np.array(T)
        T = np.reshape(T, (1, -1))
        
        return T

    def afficheinfo(self):
        """
        Returns the classification results and the closest color name.

        Returns:
            tuple: A tuple containing the classification results dictionary and the closest color name.
        """

        return self.L, self.c.couleur_proche()


class Window(QWidget):
    """
    Window class represents the main interface for displaying a video feed and processing images.

    Signals:
        ImageUpdate (QImage): Signal emitted when a new frame is available to update the displayed image.
        PrediUpdate (list): Signal emitted when a new prediction result is available.

    Attributes:
        Worker1 (Worker1): Worker1 instance for capturing frames from the camera and applying filters.
        Worker2 (Worker2): Worker2 instance for image processing and prediction.
    """

    ImageUpdate = pyqtSignal(QImage)
    PrediUpdate = pyqtSignal(list)

    def __init__(self, parent=None):
        """
        Initializes the Window object and sets up the user interface.

        Parameters:
            parent (QWidget): The parent widget. Default is None.
        """

        super().__init__(parent)
        uic.loadUi("UI/Graph_Int.ui", self)

        self.marche.clicked.connect(self.marcheCl)
        self.arret.clicked.connect(self.fin)
        self.horizontalSlider.valueChanged.connect(self.changet)
        self.horizontalSlider_2.valueChanged.connect(self.changettr1)
        self.cancelButton.clicked.connect(self.CancelFeed)
        self.cB.currentIndexChanged.connect(self.traitim)

        self.Worker1 = Worker1()
        self.Worker1.start()

        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.frameUpdate.connect(self.Worker2.pr)

        self.Worker2 = Worker2()
        self.Worker2.start()

        self.Worker2.PrediUpdate.connect(self.PrediUpdateSlot)

        self.Worker2.tab = [1, 3]
        self.Worker2.boolcouleur1 = 0
        self.Worker2.boolcouleur2 = 0
        self.Worker2.boolcouleur3 = 0
        self.Worker2.boolforme1 = 0
        self.Worker2.boolforme2 = 0
        self.Worker2.boolforme3 = 0
        self.Worker2.boolchoix1 = 0
        self.Worker2.boolchoix2 = 0
        self.Worker2.boolchoix3 = 0
        self.Worker2.cp1 = 0
        self.Worker2.cp2 = 0
        self.Worker2.cp3 = 0
        self.Worker2.tab1 = [[0, 0, 0], [0, 0, 0, 0, 0]]
        self.Worker2.tab2 = [[0, 0, 0], [0, 0, 0, 0, 0]]
        self.Worker2.tab3 = [[0, 0, 0], [0, 0, 0, 0, 0]]

        self.Worker2.detection()

    def CancelFeed(self):
        """
        Cancels the video feed and stops the workers.
        """

        self.Worker2.stop()
        self.Worker1.stop()

    def changet(self):
        """
        Slot for handling changes in the time interval.
        """

        self.Worker1.timacsses(self.horizontalSlider.value())
        self.time_label.setText("%d s" % self.horizontalSlider.value())

    def changeim(self, c):
        """
        Slot for handling changes in the image processing mode.

        Parameters:
            c: The new value of the image processing mode.
        """

        self.Worker1.trait(c)

    def changettr1(self):
        """
        Slot for handling changes in the threshold value.
        """

        self.Worker1.traitth1(self.horizontalSlider_2.value())
        self.Worker2.traitth1(self.horizontalSlider_2.value())

    def ImageUpdateSlot(self, Image):
        """
        Slot for updating the displayed image.

        Parameters:
            Image (QImage): The new image to be displayed.
        """

        self.label.setPixmap(QPixmap.fromImage(Image))

    def PrediUpdateSlot(self, pred):
        """
        Slot for updating the prediction result.

        Parameters:
            pred (list): The prediction result.
        """

        self.nbpiece1.setText(str(self.Worker2.cp1))
        self.nbpiece2.setText(str(self.cp2))
        self.nbpiece3.setText(str(self.cp3))
        self.Worker2.detection()
        self.label_3.setText(str(pred[0]))
        self.nclabel.setText("Couleur : " + str(pred[1]))
        self.clabel.setStyleSheet(
            "background-color:rgb(" + str(pred[2][0]) + "," + str(pred[2][1]) + "," + str(pred[2][2]) + ")")

    def traitim(self, val):
        """
        Slot for handling changes in the image processing mode.

        Parameters:
            val: The new value of the image processing mode.
        """

        self.Worker1.trait(val)

    def fin(self):
        """
        Slot for stopping the workers.
        """

        self.a = 1

    def marcheCl(self):
        """
        Slot for starting the workers and configuring the processing parameters.
        """

        self.Worker2.tab1 = [[0, 0, 0], [0, 0, 0, 0, 0]]

        if self.forme1.isChecked():

            if self.rond1.isChecked():
                self.Worker2.tab1[1][1] = 1
            if self.carre1.isChecked():
                self.Worker2.tab1[1][0] = 1
            if self.triangle1.isChecked():
                self.Worker2.tab1[1][2] = 1
            if self.pentagone1.isChecked():
                self.Worker2.tab1[1][3] = 1
            if self.etoile1.isChecked():
                self.Worker2.tab1[1][4] = 1

        if self.couleur1.isChecked():

            if self.rouge1.isChecked():
                self.Worker2.tab1[0][0] = 1
            if self.vert1.isChecked():
                self.Worker2.tab1[0][1] = 1
            if self.bleu1.isChecked():
                self.Worker2.tab1[0][2] = 1

        self.Worker2.tab2 = [[0, 0, 0], [0, 0, 0, 0, 0]]

        if self.forme2.isChecked():

            if self.rond2.isChecked():
                self.Worker2.tab2[1][1] = 1
            if self.carre2.isChecked():
                self.Worker2.tab2[1][0] = 1
            if self.triangle2.isChecked():
                self.Worker2.tab2[1][2] = 1
            if self.pentagone2.isChecked():
                self.Worker2.tab2[1][3] = 1
            if self.etoile2.isChecked():
                self.Worker2.tab2[1][4] = 1

        if self.couleur2.isChecked():

            if self.rouge2.isChecked():
                self.Worker2.tab2[0][0] = 1
            if self.vert2.isChecked():
                self.Worker2.tab2[0][1] = 1
            if self.bleu2.isChecked():
                self.Worker2.tab2[0][2] = 1

        self.Worker2.tab3 = [[0, 0, 0], [0, 0, 0, 0, 0]]
        self.Worker2.tabchoice3 = [0, 0]

        if self.forme3.isChecked():

            if self.rond3.isChecked():
                self.Worker2.tab3[1][1] = 1
            if self.carre3.isChecked():
                self.Worker2.tab3[1][0] = 1
            if self.triangle3.isChecked():
                self.Worker2.tab3[1][2] = 1
            if self.pentagone3.isChecked():
                self.Worker2.tab3[1][3] = 1
            if self.etoile3.isChecked():
                self.Worker2.tab3[1][4] = 1

        if self.couleur3.isChecked():

            if self.rouge3.isChecked():
                self.Worker2.tab3[0][0] = 1
            if self.vert3.isChecked():
                self.Worker2.tab3[0][1] = 1
            if self.bleu3.isChecked():
                self.Worker2.tab3[0][2] = 1

        self.m = 1


class Worker1(QThread):
    """
    A worker thread class for image processing.

    Signals:
        - ImageUpdate: A PyQt signal emitted when a processed image is ready to be displayed.
        - frameUpdate: A PyQt signal emitted when a frame is ready to be processed.

    Attributes:
        - timact (int): The time interval between processed frames in seconds.
        - ThreadActive (bool): Indicates whether the worker thread is active or not.
        - traitement (int): The type of image processing to be performed.
        - th1 (int): The first threshold value for image processing.
        - th2 (int): The second threshold value for image processing.
    """

    ImageUpdate = pyqtSignal(QImage)
    frameUpdate = pyqtSignal(np.ndarray)

    def run(self):
        """
        Starts the worker thread and performs image processing.

        This method runs in a loop, continuously capturing frames from the camera, processing them according to the
        specified image processing type, and emitting signals to update the UI with the processed frames.

        Returns:
            None
        """
        self.timact = 2
        self.ThreadActive = True
        self.traitement = 2
        self.th1 = 120
        self.th2 = 120

        # Variables
        # 0: first available camera;  1-254: The camera with the specified camera ID
        hCam = ueye.HIDS(0)
        sInfo = ueye.SENSORINFO()
        cInfo = ueye.CAMINFO()
        pcImageMemory = ueye.c_mem_p()
        MemID = ueye.int()
        rectAOI = ueye.IS_RECT()
        pitch = ueye.INT()
        # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        nBitsPerPixel = ueye.INT(24)
        # 3: channels for color mode(RGB); take 1 channel for monochrome
        channels = 3
        m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
        bytes_per_pixel = int(nBitsPerPixel / 8)
        # ---------------------------------------------------------------------------------------------------------------------------------------
        print("START")
        print()

        # Starts the driver and establishes the connection to the camera
        nRet = ueye.is_InitCamera(hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")

        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
        nRet = ueye.is_GetCameraInfo(hCam, cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

        # You can query additional information about the sensor type used in the camera
        nRet = ueye.is_GetSensorInfo(hCam, sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

        nRet = ueye.is_ResetToDefault(hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")

        # Set display mode to DIB
        nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)

        # Set the right color mode
        if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
            bytes_per_pixel = int(nBitsPerPixel / 8)
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColorMode: \t\t", m_nColorMode)
            print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
            print()

        elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            nBitsPerPixel = ueye.INT(32)
            bytes_per_pixel = int(nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColorMode: \t\t", m_nColorMode)
            print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
            print()

        elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            m_nColorMode = ueye.IS_CM_MONO8
            nBitsPerPixel = ueye.INT(8)
            bytes_per_pixel = int(nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", m_nColorMode)
            print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            m_nColorMode = ueye.IS_CM_MONO8
            nBitsPerPixel = ueye.INT(8)
            bytes_per_pixel = int(nBitsPerPixel / 8)
            print("else")

        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI,
                           rectAOI, ueye.sizeof(rectAOI))
        
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        width = rectAOI.s32Width
        height = rectAOI.s32Height

        # Prints out some information about the camera and the sensor
        print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
        print("Maximum image width:\t", width)
        print("Maximum image height:\t", height)
        print()

        # ---------------------------------------------------------------------------------------------------------------------------------------

        # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(
            hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(
            hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("Press cancel to leave the programm")

        # ---------------------------------------------------------------------------------------------------------------------------------------
        temp = time.time()
        # Continuous image display
        while (nRet == ueye.IS_SUCCESS):

            # In order to display the image in an OpenCV window we need to...
            # ...extract the data of our image memory
            array = ueye.get_data(pcImageMemory, width,
                                  height, nBitsPerPixel, pitch, copy=False)

            bytes_per_pixel = int(nBitsPerPixel / 8)

            # ...reshape it in an numpy array...
            frame = np.reshape(
                array, (height.value, width.value, bytes_per_pixel))

            # ...resize t                        he image by a half
            # frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)

        # ---------------------------------------------------------------------------------------------------------------------------------------
            # Include image data processing here

            FlippedImage = cv2.flip(frame, 1)
            FlippedImage = FlippedImage[:, :, :3]


            if self.traitement == 2:

                NBI = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2GRAY)
                _, NBI = cv2.threshold(
                    NBI, int((self.th1*255)/100), 255, cv2.THRESH_BINARY)
                NBI = cv2.cvtColor(NBI, cv2.COLOR_GRAY2BGR)
                FlippedImage = cv2.bitwise_and(FlippedImage, NBI)

            if self.traitement == 1:

                FlippedImage = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2GRAY)

            if self.traitement == 3:

                FlippedImage = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2HSV)
                hsv = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                # otsu choisi un trehol a la moitier des valeurs de l'histogramme
                ret_h, th_h = cv2.threshold(
                    h, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                ret_s, th_s = cv2.threshold(
                    s, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                # Fusion th_h et th_s
                th = cv2.bitwise_or(th_h, th_s)
                # Ajouts de bord à l'image
                bordersize = 50
                th = cv2.copyMakeBorder(th, top=bordersize, bottom=bordersize, left=bordersize,
                                        right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
                # Remplissage des contours
                im_floodfill = th.copy()
                h, w = th.shape[:2]
                mask = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(im_floodfill, mask, (0, 0), 255)
                im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                th = th | im_floodfill_inv
                # Enlèvement des bord de l'image
                th = th[bordersize: len(th)-bordersize,
                        bordersize: len(th[0])-bordersize]
                resultat = cv2.bitwise_and(FlippedImage, FlippedImage, mask=th)
                # cv2.imwrite("im_floodfill.png",im_floodfill)
                # cv2.imwrite("th.png",th)
                # cv2.imwrite("resultat.png",resultat)
                # FlippedImage=cv2.merge((th_h,th_s,th_v))
                FlippedImage = resultat
                FlippedImage = cv2.cvtColor(FlippedImage, cv2.COLOR_HSV2BGR)

            if self.traitement == 4:

                # FlippedImage = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(FlippedImage, (5, 5), 0)
                # thresh=cv2.Canny(FlippedImage, self.th1,self.th1*2)
                thresh = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
                FlippedImage = thresh

            if temp+self.timact < time.time():

                updImage = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2RGB)
                # Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # FlippedImage = cv2.flip(Image, 1)
                # npim=np.array(FlippedImage).astype(np.uint8)
                # imia=np.concatenate([npim,tz])
                # np.save("/Users/alexis/Desktop/imiai", imia)
                # print("okokokokokokokokok")
                # self.prediction=b.classification(Frame=imia)
                # print(type(FlippedImage))
                self.frameUpdate.emit(updImage)
                temp += self.timact

            if self.traitement == 6:

                hsv = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                ret_h, th_h = cv2.threshold(
                    h, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                ret_s, th_s = cv2.threshold(
                    s, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                hsv = cv2.merge((th_h, th_s, v))
                brgImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                gray = cv2.cvtColor(brgImage, cv2.COLOR_BGR2GRAY)
                #  apply thresholding on the gray image to create a binary image
                _, thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                # gray = cv2.GaussianBlur(gray, (5, 5), 0)
                # thresh=cv2.Canny(FlippedImage, self.th1,self.th1*2)
                # find the contours
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for i in range(len(contours)):

                    # take the first contour
                    cnt = contours[i]
                    # compute the bounding rectangle of the contour
                    x, y, w, h = cv2.boundingRect(cnt)
                    # draw contour
                    img = cv2.drawContours(FlippedImage, [
                                           cnt], contourIdx=-1, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                    # thresh=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
                    # img = cv2.drawContours(thresh,[cnt],0,(0,255,255),2)
                    # draw the bounding rectangle
                    img = cv2.rectangle(
                        img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                FlippedImage = img

            if self.traitement == 5:

                FlippedImage = cv2.GaussianBlur(FlippedImage, (5, 5), 0)
                FlippedImage = cv2.Canny(FlippedImage, self.th1, self.th1*2)

        # ---------------------------------------------------------------------------------------------------------------------------------------

            # ...and finally display it


            FlippedImage = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2RGB)
            ConvertToQtFormat = QImage(
                FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)  # Format_Mono)#
            # 640, 480, Qt.KeepAspectRatio)
            
            Pic = ConvertToQtFormat.scaled(640, 512, Qt.KeepAspectRatio)
            
            # cree le signal qui permet d'actualiser les data
            self.ImageUpdate.emit(Pic)
            
            # Press q if you want to end the loop
            
            if self.ThreadActive == False:
                break

        # ---------------------------------------------------------------------------------------------------------------------------------------

        # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
        ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)

        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(hCam)

        # Destroys the OpenCv windows
        # cv2.destroyAllWindows()

        print()
        print("END")

    def stop(self):
        """
        Stop the execution of the class.

        This method sets the 'ThreadActive' flag to False and quits the execution of the class.
        """
        self.ThreadActive = False
        self.quit()

    def timacsses(self, ti):
        """
        Set the value of 'timact' attribute.

        Parameters:
        ti (float): The value to set for 'timact'.
        """
        self.timact = ti

    def trait(self, t):
        """
        Set the value of 'traitement' attribute.

        Parameters:
        t (str): The value to set for 'traitement'.
        """
        self.traitement = t

    def traitth1(self, t):
        """
        Set the value of 'th1' attribute.

        Parameters:
        t (int): The value to set for 'th1'.
        """
        self.th1 = t

    def traitth2(self, t):
        """
        Set the value of 'th2' attribute.

        Parameters:
        t (int): The value to set for 'th2'.
        """
        self.th2 = t


class Worker2(QThread):
    """
    A worker class for performing tasks in a separate thread.

    This class inherits from QThread and is designed to perform certain operations in the background.
    It utilizes the IAcouleurs class for image color representation and shape classification using a neural network.
    The goal of this class is to process images, perform classification, and send classification information to a nuclei card
    to activate motors.
    """
    PrediUpdate = pyqtSignal(list)

    def __init__(self):
        """
        Initialize the Worker2 class.

        This method sets up the initial state and configurations for the Worker2 class.
        It allows the user to choose a port for communication with the card that is connected to the computer.

        The Worker2 class initializes various attributes such as image arrays, a color and shape classification tool
        represented by the 'IAcouleurs' class, threshold values, and lists for storing predictions.

        After a delay of 10 seconds (time for the other thread to start, so it needs to be increased if your computer runs slow\...), it retrieves the available serial ports and prompts the user to select a COM port.
        Once the port is selected, a serial connection is established with the specified port at a baud rate of 115200.

        """
        super().__init__()
        
        self.tz = np.zeros([304, 1280, 3]).astype(np.uint8)
        self.b = IAcouleurs()
        self.th1 = 120
        self.tab = [0, 0]
        self.L = []
        self.clf = []

        time.sleep(10)
        
        ports = serial.tools.list_ports.comports()
        
        for port, desc, hwid in sorted(ports):
            
            print("{}: {}".format(port, desc))
            selectPort = input("Select a COM port : ")
            
        print(f"Port Selected : COM{selectPort}")
        
        self.serNuc = Serial('COM'+str(selectPort), 115200)

    def pr(self, Imagecv):
        """
        Perform image processing and prediction.

        This method receives an image, performs image processing and prediction using the 'classification' and 'classi2'
        methods of the 'IAcouleurs' class. It emits the prediction through the 'PrediUpdate' signal.

        The 'IAcouleurs' class represents an image color and a shape classification tool using a neural network.

        The goal of this function is to send the information of the classification to a nuclei card to activate motors.
        It determines the most likely prediction by comparing several frame predictions during the passage of an object
        in front of the camera.

        Parameters:
        Imagecv (cv2.Image): The input image for prediction.
        """

        npim = np.array(Imagecv).astype(np.uint8)
        imia = npim
        
        self.prediction = self.b.classification(
            Frame=imia, s=int((self.th1*255)/100))
        
        self.PrediUpdate.emit(self.prediction)

        maxcoul = self.prediction[2][0]
        
        for k in range(len(self.prediction[2])):
            
            if self.prediction[2][k] >= maxcoul:
                couleur_rgb_finale = k

        pred = self.b.classi2(Frame=imia, s=int((self.th1*255)/100))
        
        if pred != -1:
            
            self.clf.append(couleur_rgb_finale)
            self.L.append(pred[0])

        elif len(self.L) > 1:
            
            prL = [0, 0, 0, 0, 0]
            
            for k in range(len(self.L)):
                prL[self.L[k]] += 1
                
            prL = np.array(prL)
            pre = prL.argmax()
            
            self.tab = [self.clf[-2], pre]
            print("prediction :")
            print(self.tab)
            self.L = []
            self.clf = []

    def stop(self):
        """
        Stop the Worker2 class execution.

        This method closes the serial connection with the card and stops the execution of the Worker2 class.

        """
        self.serNuc.close()
        self.quit()  # Stop class execution

    def traitth1(self, t):

        self.th1 = t

    def detection(self):
        """
        Process the detection results and send commands to activate motors based on the classification.

        This method checks the detection results stored in `tab1`, `tab2`, and `tab3` lists. 
        If any of the lists contains non-zero values, it determines the corresponding action to take based on the values.
        If the values in `tab1` indicate a classification result, it sends a command to activate motor1.
        If the values in `tab2` indicate a classification result, it sends a command to activate motor2.
        If the values in `tab3` indicate a classification result, it sends a command to activate motor3.

        """
        if self.tab1[0] != [0, 0, 0] or self.tab1[1] != [0, 0, 0, 0, 0]:
            
            if self.tab1[1] == [0, 0, 0, 0, 0]:
                if self.tab1[0][self.tab[0]] == 1:  # /!\-1?
                    self.serNuc.write(bytes(str(1), 'ascii')
                                      )  # commande moteur1
                    print("Send 1")
                    # while self.serNuc.inWaiting() == 0:
                    #     pass
                    # data_rec = self.serNuc.read(4)  # bytes
                    # print(str(data_rec))
                    self.cp1 += 1  # compteur de piece
                    # self.nbpiece1.setText(str(self.cp1))
                    
            if self.tab1[0] == [0, 0, 0]:
                if self.tab1[1][self.tab[1]] == 1:  # /!\-1?
                    self.serNuc.write(bytes(str(1), 'ascii')
                                      )  # commande moteur1
                    print("Send 1")
                    # while self.serNuc.inWaiting() == 0:
                    #     pass
                    # data_rec = self.serNuc.read(4)  # bytes
                    # print(str(data_rec))
                    self.cp1 += 1  # compteur de piece
                    # self.nbpiece1.setText(str(self.cp1))
                    
            else:
                
                if self.tab1[0][self.tab[0]] == 1:  # -1?
                    if self.tab1[1][self.tab[1]] == 1:
                        # commande moteur1
                        self.serNuc.write(bytes(str(1), 'ascii'))

                        # while self.serNuc.inWaiting() == 0:
                        #     pass
                        # data_rec = self.serNuc.read(4)  # bytes
                        # print(str(data_rec))
                        self.cp1 += 1  # compteur de piece
                        # self.nbpiece1.setText(str(self.cp1))

        if self.tab2[0] != [0, 0, 0] or self.tab2[1] != [0, 0, 0, 0, 0]:
            
            if self.tab2[1] == [0, 0, 0, 0, 0]:
                if self.tab2[0][self.tab[0]] == 1:  # /!\-1?
                    self.serNuc.write(bytes(str(1), 'ascii')
                                      )  # commande moteur1

                    # while self.serNuc.inWaiting() == 0:
                    #     pass
                    # data_rec = self.serNuc.read(4)  # bytes
                    # print(str(data_rec))
                    self.cp1 += 1  # compteur de piece
                    # self.nbpiece1.setText(str(self.cp1))
                    
            if self.tab2[0] == [0, 0, 0]:
                if self.tab2[1][self.tab[1]] == 1:  # /!\-1?
                    self.serNuc.write(bytes(str(1), 'ascii')
                                      )  # commande moteur1

                    # while self.serNuc.inWaiting() == 0:
                    #     pass
                    # data_rec = self.serNuc.read(4)  # bytes
                    # print(str(data_rec))
                    self.cp1 += 1  # compteur de piece
                    # self.nbpiece1.setText(str(self.cp1))
                    
            else:
                if self.tab2[0][self.tab[0]] == 1:  # -1?
                    if self.tab2[1][self.tab[1]] == 1:
                        # commande moteur1
                        self.serNuc.write(bytes(str(1), 'ascii'))
                        # while self.serNuc.inWaiting() == 0:
                        #     pass
                        # data_rec = self.serNuc.read(4)  # bytes
                        # print(str(data_rec))
                        self.cp1 += 1  # compteur de piece
                        # self.nbpiece1.setText(str(self.cp1))

        if self.tab3[0] != [0, 0, 0] or self.tab3[1] != [0, 0, 0, 0, 0]:
            
            if self.tab3[1] == [0, 0, 0, 0, 0]:
                if self.tab3[0][self.tab[0]] == 1:  # /!\-1?
                    self.serNuc.write(bytes(str(1), 'ascii')
                                      )  # commande moteur1
                    print("envoyé3")
                    # while self.serNuc.inWaiting() == 0:
                    #     pass
                    # data_rec = self.serNuc.read(4)  # bytes
                    # print(str(data_rec))
                    self.cp1 += 1  # compteur de piece
                    # self.nbpiece1.setText(str(self.cp1))
                    
            if self.tab3[0] == [0, 0, 0]:
                if self.tab3[1][self.tab[1]] == 1:  # /!\-1?
                    self.serNuc.write(bytes(str(1), 'ascii')
                                      )  # commande moteur1
                    print("envoyé3")
                    # while self.serNuc.inWaiting() == 0:
                    #     pass
                    # data_rec = self.serNuc.read(4)  # bytes
                    # print(str(data_rec))
                    self.cp1 += 1  # compteur de piece
                    # self.nbpiece1.setText(str(self.cp1))
                    
            else:
                if self.tab3[0][self.tab[0]] == 1:  # -1?
                    if self.tab3[1][self.tab[1]] == 1:
                        # commande moteur1
                        self.serNuc.write(bytes(str(1), 'ascii'))
                        # while self.serNuc.inWaiting() == 0:
                        #     pass
                        # data_rec = self.serNuc.read(4)  # bytes
                        # print(str(data_rec))
                        self.cp1 += 1  # compteur de piece
                        # self.nbpiece1.setText(str(self.cp1))


if __name__ == "__main__":

    app = QApplication(sys.argv)

    win = Window()
    win.show()
    app.exec()
