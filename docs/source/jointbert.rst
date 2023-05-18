JointBERT Evaluation
====================

Environment Set-up
------------------

In order to install dependencies to test the JointBERT model use the following command. 

.. note:: 
   Quotes are needed around the path to ensure JointBERT dependencies are installed

.. code-block:: console

   $ pip install '/path/to/python/wheel.whl[jointbert]'

Evaluation
----------

Once installed JointBERT model can be evaluated using the following command. 

.. code-block:: console

    $ python examples/evaluate_jointbert.py --save_dir /path/to/results/dir

This will generate an HTML robustness report on the SNIPS dataset
