import unittest

import spacy
import numpy as np

from auditor.perturbations import PerturbText
from auditor.evaluation.evaluate import ModelTest, TestSuite
from auditor.evaluation.expected_behavior import InvariantScore
from .validation_utils import get_ner_pipeline


TEST_DATASET = [
    "please call michael",
    "please call michael bolton",
    "how's the weather in Austin",
    "Set timer for 5 minutes",
]

def model_predict(input_text):
    print(f'Input text: {input_text}')
    out_len = 2
    inp_len = len(input_text)
    return np.ones((inp_len, out_len)) 

class TestModelEval(unittest.TestCase):
    def setUp(self) -> None:
        # ner_pipeline = spacy.load("en_core_web_sm").pipe
        ner_pipeline = get_ner_pipeline()
        self.perturber = PerturbText(
            TEST_DATASET,
            ner_pipeline=ner_pipeline,
            batch_size=8,
            perturbations_per_sample=5,
        )
        return

    def test_model_eval(self):
        perturbed_dataset = self.perturber.perturb_names()
        invariant_behavior = InvariantScore()
        model_eval = ModelTest(
            perturbed_dataset=perturbed_dataset,
            expected_behavior=invariant_behavior,
        )
        test_details = model_eval.evaluate(model_predict=model_predict)
        print(test_details)
        return
    
    def test_model_suite(self):
        test_suite = TestSuite(
            model_predict=model_predict,
            description='Test-suite for dummy model with perturbed names and paraphrasing'
        )

        # test 1
        perturbed_names = self.perturber.perturb_names()
        invariant_behavior = InvariantScore()
        test_names = ModelTest(
            perturbed_dataset=perturbed_names,
            expected_behavior=invariant_behavior,
        )
        
        test_suite.add(test_names)

        # test 2
        paraphrased_dataset = self.perturber.paraphrase()
        invariant_behavior = InvariantScore()
        test_paraphrase = ModelTest(
            perturbed_dataset=paraphrased_dataset,
            expected_behavior=invariant_behavior,
        )
        test_suite.add(test_paraphrase)

        # run the suite
        suite_summary = test_suite.evaluate()

        # generate report
        test_suite.generate_html_report(
            suite_summary=suite_summary,
            model_name='test model',
            save_dir='/tmp/simple_model_robustness/',
        )   