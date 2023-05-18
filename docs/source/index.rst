.. fiddler-modelauditor documentation master file, created by
   sphinx-quickstart on Mon Dec  5 12:16:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


================================================
Welcome to fiddler-auditor's documentation!
================================================

.. .. image:: images/architecture.png
..    :width: 800px
..    :align: center


**fiddler-auditor** provides a set of tools to audit Generative and Discriminative NLP models.
It aims to provide a flexible framework that can perturb data, evaluate model and generate reports.

Installation
============

To use fiddler-auditor,
first create a new python environment and install the python wheel using pip.

.. code-block:: console

   $ conda create -n audit-llm
   $ conda activate audit-llm
   $ pip install fiddler-auditor

.. toctree::
   :caption: User Guide
   :maxdepth: 1
   
   Quick Start <LLM_Evaluation>
   API Reference <modules>