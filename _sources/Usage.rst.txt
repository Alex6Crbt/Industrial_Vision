Usage
=====


.. caution::
   This website is still a work in progress


To use the Industrial_Vision library for shape recognition and sorting automation, 
you need to follow the installation steps outlined below.

Installation
------------

1. **Prerequisites**\: Ensure that you have Python 3.x installed on your system. 
If you don't have Python installed, you can download it from the official Python 
website (https://www.python.org).

2. **Create a Virtual Environment (Optional)**\: It is recommended to create a 
virtual environment to keep the project dependencies isolated. Open a terminal 
or command prompt and run the following commands\:

   On macOS and Linux\:
   
.. code-block:: console
    
    python3 -m venv myenv
    source myenv/bin/activate
       
    
   On Windows\:
   
.. code-block:: console

    python -m venv myenv
    myenv\Scripts\activate
   

3. **Install Required Packages**\: In the activated virtual environment or your 
global Python environment, run the following command to install the necessary packages\:

.. code-block:: console

    pip install scikit-learn opencv-python

   This command installs the scikit-learn library, which is used for training 
   the AI model, and the OpenCV library (opencv-python), which provides computer 
   vision functionalities.

4. **Download Industrial_Vision Library**\: Download the Industrial_Vision library 
from the project repository or source. You can either clone the repository or 
download the source code as a ZIP file.

   - If you are using Git, clone the repository by running the following command
     in your terminal or command prompt\:
    
.. code-block:: console

    git clone https://github.com/username/project-name.git
     

   - If you downloaded the ZIP file, extract it to a directory of your choice.

5. **Importing the Library**\: Once the library is downloaded, navigate to the 
project directory and import it into your Python script or interactive session 
using the following import statement\:

.. code-block:: python

    import Industrial\_Vision
   

6. **Initializing and Utilizing the Library**\: Follow the documentation or 
examples provided with the Industrial_Vision library to initialize the camera 
system, configure the sorting mechanism with servo motors, and utilize the shape 
recognition capabilities.

7. **Running the Application**\: Execute your Python script or interact with the 
Industrial_Vision library using the appropriate commands and functions as described 
in the library's documentation.

Conclusion
----------

By following the steps outlined in this Usage guide, you can successfully install 
the Industrial_Vision library and begin utilizing its capabilities for shape 
recognition and sorting automation. Ensure that you have the necessary prerequisites 
and dependencies installed, and don't forget to consult the library's documentation 
for further guidance on initialization, configuration, and usage.

