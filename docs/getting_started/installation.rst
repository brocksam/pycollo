==========================
Pycollo Installation Guide
==========================

Prerequisites
=============

Before you begin, ensure you have the following prerequisites installed:

- Python (>= 3.7)

Step 1: Create a Virtual Environment
------------------------------------

It is recommended to install Pycollo in a virtual environment to manage dependencies.

1. Open a terminal or command prompt.

2. Create a new virtual environment:
   
   .. code-block:: bash
      
      python -m venv pycollo-env

3. Activate the virtual environment:

   - On Windows:

     .. code-block:: bash
     
       .\pycollo-env\Scripts\activate

   - On macOS and Linux:

     .. code-block:: bash
     
        source pycollo-env/bin/activate

Step 2: Install Pycollo
------------------------
To install with conda-forge enter the following command at a command prompt

.. code-block:: bash

    conda install -c conda-forge pycollo

To install using conda, enter the following command at a command prompt:

.. code-block:: bash

    conda install pycollo

To install using pip, enter the following command at a command prompt:

.. code-block:: bash

    pip install pycollo


Pycollo and its dependencies will be downloaded and installed.

.. note::

   If you no longer need the virtual environment, you can deactivate it by running the command:

   .. code-block:: bash
   
      deactivate