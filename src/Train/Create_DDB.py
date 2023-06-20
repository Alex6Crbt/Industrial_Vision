import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import pandas as pd


class CreeBD:
    """
    Class for creating the image database.

    Parameters:
    - Test (bool): Indicates whether to run the create test database mode or not. Default is True.
    - nbrimageset (int): Number of image sets to generate. Default is 500.
    """

    def __init__(self, Test=True, nbrimageset=500):
        """
        Initializes the CreeBD class.

        Args:
        - Test (bool): Indicates whether to run the test mode or not. Default is True.
        - nbrimageset (int): Number of image sets to generate. Default is 500.
        """

        if Test:
            print('Loading and processing images')
            img = Image.open('images/a8.bmp')
            img = img.convert('L')
            img = np.array(img)
            (n, p) = img.shape

            # Parameters
            # nbrimageset=500
            # conversion of X (list de table(image)) to Xl (list of list(images in 1 list))
            print("Create random images")

            Xl = []
            X = self.randgen(img, nbrimageset)

            for k in range(nbrimageset):
                V = []
                for i in range(51):  # /!\51!!!!
                    V += ((X[k]).tolist())[i]
                Xl.append(V)

            print("Generation of the database in .csv")

            # /!\ Be careful nbrimageset=3500 or 500?
            data = {str(i): Xl[i] for i in range(nbrimageset)}
            df = pd.DataFrame(data)
            df.to_csv(r'BaseEtTEST.csv', index=True, header=True)

            print(df)
        else:
            print('Loading images')
            img1 = Image.open('images/t1.bmp')
            img2 = Image.open('images/t2.bmp')
            img3 = Image.open('images/t3.bmp')
            img4 = Image.open('images/t4.bmp')
            img5 = Image.open('images/t5.bmp')
            img6 = Image.open('images/t6.bmp')
            img7 = Image.open('images/t7.bmp')

            img1 = img1.convert('L')
            img1 = np.array(img1)
            img2 = img2.convert('L')
            img2 = np.array(img2)
            img3 = img3.convert('L')
            img3 = np.array(img3)
            img4 = img4.convert('L')
            img4 = np.array(img4)
            img5 = img5.convert('L')
            img5 = np.array(img5)
            img6 = img6.convert('L')
            img6 = np.array(img6)
            img7 = img7.convert('L')
            img7 = np.array(img7)

            LIMG = [img1, img2, img3, img4, img5, img6, img7]

            # conversion X (liste de tableau(image)) en Xl (liste de liste(image en 1 liste))
            print("Create random images")
            Xl = []
            for j in tqdm(LIMG):
                X = self.randgen(j, nbrimageset, False)
                for k in range(nbrimageset):
                    V = []
                    for i in range(51):  # /!\51!!!!
                        V += ((X[k]).tolist())[i]
                    Xl.append(V)

            # création de la base de data en .csv
            print("Creation of the database in .csv")

            # /!\ Be careful nbrimageset=3500 or 500
            data = {str(i): Xl[i] for i in range(nbrimageset*7)}
            df = pd.DataFrame(data)

            print('Convert to .csv')

            df.to_csv(r'BaseTri2.csv', index=True, header=True)

            print(df)

    def randgen(self, im, ni, tq=True):
        """
        Generates random images.

        Args:
        - im: Input image.
        - ni (int): Number of images to generate.
        - tq (bool): Indicates whether to display progress bar. Default is True.

        Returns:
        - X (list): List of generated images.
        """

        if tq:
            X = [0] * ni
            r = np.random.randn(ni, 3)
            im = self.seuil(im, 120)
            im = self.resize(im, 64, 64)

            for s in tqdm(range(0, ni)):
                iml = self.rotation(im, r[s, 0] * 360)
                iml = self.translation(iml, r[s, 1] * 10, r[s, 2] * 10)
                X[s] = iml

            return X
        else:
            X = [0] * ni
            r = np.random.randn(ni, 3)
            im = self.seuil(im, 120)
            im = self.resize(im, 64, 64)

            for s in range(0, ni):
                iml = self.rotation(im, r[s, 0] * 360)
                iml = self.translation(iml, r[s, 1] * 10, r[s, 2] * 10)
                X[s] = iml

            return X

    def translation(self, image, x, y):
        """
        Translates the image.

        Args:
        - image: Input image.
        - x (float): Translation along the x-axis.
        - y (float): Translation along the y-axis.

        Returns:
        - Translated image.
        """
        return ndimage.shift(image, (x, y))

    def rotation(self, image, deg):
        """
        Rotates the image.

        Args:
        - image: Input image.
        - deg (float): Rotation angle in degrees.

        Returns:
        - Rotated image.
        """
        return ndimage.rotate(image, deg, reshape=False)

    def resize(self, image, l, m):
        """
        Resizes the image.

        Args:
        - image: Input image.
        - l (int): Width of the resized image.
        - m (int): Height of the resized image.

        Returns:
        - Resized image.
        """
        image = Image.fromarray(np.uint8((image)))
        image.thumbnail((l, m))  # /!\.shape=(51,64)

        return np.array(image)

    def seuil(self, imNB, s):
        """
        Applies a threshold to the image.

        Args:
        - imNB: Input grayscale image.
        - s (int): Threshold value.

        Returns:
        - Thresholded image.
        """
        return np.where(imNB < s, 0, imNB)
