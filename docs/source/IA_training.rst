IA\_training module
===================

The EntrainementIA algorithm is designed to train an artificial neural network, specifically a Multi-Layer Perceptron (MLP), using the preprocessed databases.

.. dropdown:: **Features** :octicon:`flame`  
    :open:
    :animate: fade-in-slide-down

    The algorithm begins by importing the preprocessed databases, which contain data in the form required for training the MLP. These databases include different shapes such as squares, circles, triangles, pentagons, and stars. The data from each shape is organized and combined into a single dataset.

    Next, the algorithm sets up the training and testing data witch will be used to evaluate the trained model's performance.

    The MLP classifier is configured with specific hyperparameters, such as the number and size of hidden layers, activation function, learning rate, and optimization algorithm. These parameters can be adjusted to optimize the model's performance.

    The algorithm then proceeds to train the MLP model using the training data. During training, the MLP learns to classify the input images into their respective shape categories based on the provided target labels.

    After training, the algorithm provides various information about the trained model. It displays the accuracy score, indicating how well the model performs on the testing data. The loss curve is also shown, which represents the training loss at each iteration, providing insights into the model's convergence.

    The algorithm includes additional functionalities, such as displaying individual images from the datasets, visualizing weight matrices of the trained model, and generating a confusion matrix to evaluate the model's performance in detail.

.. tip::

   The algorithm also provides the ability to save the trained model's weights for future use. 
   It also allows loading previously saved weights to set the MLP model's parameters.



.. automodule:: IA_training
   :members:
   :special-members: __init__
   :show-inheritance:
