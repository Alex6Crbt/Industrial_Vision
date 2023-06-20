from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


print("librairies imported",end='\n')



class EntrainementIA:
    """
    Class for training an artificial neural network (MLP) using the preprocessed databases.

    Attributes:
        x_train (ndarray): Input features for training.
        y_train (ndarray): Target labels for training.
        x_test (ndarray): Input features for testing.
        y_test (ndarray): Target labels for testing.
        mlp (MLPClassifier): Multi-layer Perceptron classifier.

    """

    def __init__(self):
        """
        Initialize the EntrainementIA class.

        This method imports the preprocessed databases, sets up the training and testing data,
        configures the MLP classifier, and performs the training.

        Note:
        - The preprocessed databases should be stored in CSV files with specific filenames.
        - The MLP classifier hyperparameters can be adjusted within the method.
        """

        # Importing and setting up the training data
        print("Importing databases:", end='\n * ')
        dfA = pd.read_csv(r'BaseCarre2.csv')
        a1 = [dfA[str(i)].values for i in range(17500)]
        # ... (similar steps for other databases)

        # Combining the training data
        x = a1 + o1 + t1 + p1 + e1 + N
        x = np.array(x)

        # Setting up the target labels
        y = [0] * (17500) + [1] * (17500) + [2] * 17500 + [3] * 17500 + [4] * 17500 + [-1] * 3500
        y = np.array(y)

        self.x_train = x
        self.y_train = y

        # Setting up the testing data
        # ... (similar steps as above)

        self.x_test = xt
        self.y_test = yt

        # Setting up the MLP classifier
        self.mlp = MLPClassifier(hidden_layer_sizes=([256, 128, 128]), activation='logistic',
                                 alpha=1e-4, solver='sgd', tol=5e-3, random_state=None,
                                 verbose=True, max_iter=200, warm_start=False, learning_rate_init=0.005)

        # Training the MLP model
        self.Fit()
        self.afficheinfo()
        self.Impoids()
        self.matconf()

    def Fit(self):
        """
        Train the MLP model.

        This method trains the MLP model using the configured training data and target labels.
        """

        print("Training the model...")
        self.mlp.fit(self.x_train, self.y_train)

    def afficheinfo(self):
        """
        Display information about the trained model.

        This method displays the accuracy score and loss curve of the trained model.
        """

        predi = self.mlp.predict(self.x_test)
        print(accuracy_score(self.y_test, predi))
        plt.plot(self.mlp.loss_curve_)
        plt.show()

        # ... (additional plots)

    def montreimage(self, k, set):
        """
        Display the k-th image from the specified dataset.

        Args:
            k (int): Index of the image to display.
            set (ndarray): Dataset to retrieve the image from.

        Returns:
            int: Always returns 0.
        """

        X = np.reshape(set[k], (51, 64))
        imgplot = plt.imshow(X, cmap='Greys_r')
        plt.show()
        return 0

    def Impoids(self):
        """
        Display weight matrices.

        This method displays the weight matrices of the trained model.
        """

        print("Training set score: %f" % self.mlp.score(self.x_train, self.y_train))
        print("Test set score: %f" % self.mlp.score(self.x_test, self.y_test))

        # ... (display weight matrices)

    def matconf(self):
        """
        Display the confusion matrix.

        This method displays the confusion matrix of the trained model.
        """

        y_pred = self.mlp.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(cm).plot()
        plt.show()

    def sauvgarde(self):
        """
        Save the trained model's weights.

        This method saves the weight matrices and biases of the trained model.
        """

        A = self.mlp.coefs_
        B = self.mlp.intercepts_
        np.save("Poids", A)
        np.save("Biais", B)

    def setparam(self):
        """
        Set the weights of the MLP model to previously saved weights.

        This method sets the weight matrices and biases of the MLP model
        to previously saved weights.
        """

        x = np.zeros((5, 3264))
        y = [0] + [1] + [2] + [3] + [4]
        y = np.array(y)

        self.mlp.fit(x, y)

        A = np.load("Poids80k.npy", allow_pickle=True)
        A = [A[0], A[1], A[2]]
        B = np.load("Biais80k.npy", allow_pickle=True)
        B = [B[0], B[1], B[2]]

        self.mlp.coefs_ = A
        self.mlp.intercepts_ = B
