.. Mosaic documentation master file, created by
   sphinx-quickstart on Sun Apr 26 22:27:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MOSAIC
=======

*Monte-Carlo Tree Search for Algorithm Configuration* (Mosaic) is a python
library for pipeline optimization using MCTS algorithm.


.. toctree::
   :maxdepth: 2

   index

Installation
-------------
Requirements:

* Python >= 3.5.6
* pygraphviz: necessary to generate dot image files (optional)

.. code-block:: bash

   conda install graphviz
   pip install pygraphviz


Install via Github:

.. code-block:: bash

   pip install git+https://github.com/herilalaina/mosaic



Example of usage
----------------

A simple example of using **mosaic** to configure machine
learning pipeline made with PCA and SVM classifier.

.. code-block:: bash

   python examples/machine_learning.py

API
----

Search module
~~~~~~~~~~~~~

.. autoclass:: mosaic.mosaic.Search
   :members:


Native MCTS module
~~~~~~~~~~~~~~~~~~

.. autoclass:: mosaic.mosaic.MCTS
   :members:

Citing Mosaic
---------------


If you are using **mosaic** in a academic presentation, we would appreciate citation

.. code-block:: bash

   @inproceedings{ijcai2019-457,
     title     = {Automated Machine Learning with Monte-Carlo Tree Search},
     author    = {Rakotoarison, Herilalaina and Schoenauer, Marc and Sebag, Michèle},
     booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
                  Artificial Intelligence, {IJCAI-19}},
     publisher = {International Joint Conferences on Artificial Intelligence Organization},
     pages     = {3296--3303},
     year      = {2019},
     month     = {7},
     doi       = {10.24963/ijcai.2019/457},
     url       = {https://doi.org/10.24963/ijcai.2019/457},
   }
