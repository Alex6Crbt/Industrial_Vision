Quickstart
==========


.. caution::
   This website is still a work in progress


To use the Industrial_Vision library please follow the installation steps outlined below.

Installation
------------

.. dropdown:: 1. **Prerequisites**   :octicon:`code`
    :animate: fade-in-slide-down

    Ensure that you have Python 3.9 installed on your system. 
    If you don't have Python installed, you can download it from the official Python 
    website (https://www.python.org).



.. dropdown:: 2. **Create a Virtual Environment (Optional)** :octicon:`code`
    :animate: fade-in-slide-down

    It is recommended to create a 
    virtual environment to keep the project dependencies isolated. Open a terminal 
    or command prompt and run the following commands\:


    .. tab-set::

        .. tab-item:: On macOS and Linux\:

            .. code-block:: console
        
                python3 -m venv myenv
                source myenv/bin/activate

        .. tab-item:: On Windows\:

            .. code-block:: console

                python -m venv myenv
                myenv\Scripts\activate




   
.. dropdown:: 3. **Install Required Packages** :octicon:`code`
    :open:
    :animate: fade-in-slide-down
       
    In the activated virtual environment or your 
    global Python environment, run the following command to install the necessary packages\:

    .. code-block:: console

        pip install -r requirements.txt


.. dropdown:: 4. **Download Industrial_Vision Library** :octicon:`code`
    :open:
    :animate: fade-in-slide-down

    Download the Industrial_Vision library 
    from the project repository or source. You can either clone the repository or 
    download the source code as a ZIP file.

    - If you are using Git, clone the repository by running the following command
         in your terminal or command prompt\:
        
    .. code-block:: console

        git clone https://github.com/username/project-name.git
         

    - If you downloaded the ZIP file, extract it to a directory of your choice.

.. dropdown:: 5. **Importing the Library** :octicon:`code`
    :open:
    :animate: fade-in-slide-down

    Once the library is downloaded, navigate to the 
    project directory and import it into your Python script or interactive session 
    using the following import statement\:

    .. code-block:: python

        import Industrial_Vision
       

.. tip::

    Now that you have all the necessary prerequisites 
    and dependencies installed, and don't forget to consult the library's documentation 
    for further guidance on usage and the different algorithms.

