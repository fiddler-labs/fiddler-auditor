API Reference 
=============

The main classes comprise **fiddler-auditor** are as follows

* Perturbations
* Model Evaluation (LLMEval, ModelTest and TestSuite)
* Expectation Functions


Generative Model Evaluation
===========================

.. autoclass:: auditor.evaluation.evaluate.LLMEval
      :members: __init__,evaluate_prompt_robustness

Discriminative Model Evaluation
===============================

.. autoclass:: auditor.evaluation.evaluate.ModelTest
      :members: __init__, evaluate
   
.. autoclass:: auditor.evaluation.evaluate.TestSuite
      :members: __init__, add, evaluate, generate_html_report

Perturbations
=============
Perturbations that can be applied to datasets

.. autoclass:: auditor.perturbations.PerturbText
      :members: __init__,
                perturb_names,
                perturb_location,
                perturb_number,
                paraphrase