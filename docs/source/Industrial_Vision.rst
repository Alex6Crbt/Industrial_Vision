Industrial\_Vision module
=========================

This code package provides a Qt user interface class, along with two worker classes\: 


Worker1 for live video feed and image processing, and Worker2 for shape and color recognition, powered by the :py:class:`IAcouleurs` class, itself poxered by :py:class:`IAFormes` 
and :py:class:`Couleurim` class.


.. dropdown:: 1. **Features**   :octicon:`flame`
    :open:
    :animate: fade-in-slide-down


    1. **Qt User Interface Class**: The Qt user interface class provides an interactive and intuitive graphical interface for the Industrial Vision system. It enables users to interact with the system, configure settings, visualize real-time video feed, send information to the Nucleo card, and receive the recognition results.

    2. **Worker1 Class**: The Worker1 class is responsible for capturing live video feed and performing image processing operations. This class plays a vital role in preprocessing the images to ensure accurate and reliable shape and color recognition.

    3. **Worker2 Class**: The Worker2 class combines the power of AI models with shape and color recognition. It utilizes IAcouleur Class to perform analysis on preprocessed images.

    4. **IAcouleurs**: The :py:class:`IAcouleurs` Class is a combination of `IAFormes` and  `Couleurim` class that utilizes the trained machine learning model to recognize various shapes. it is also designed to identify colors in the images. It employs least squares to recognize and classify different colors with high accuracy.


.. automodule:: Industrial_Vision
   :members:
   :special-members: __init__
   :show-inheritance:
