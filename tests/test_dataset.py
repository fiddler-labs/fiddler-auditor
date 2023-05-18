import unittest

import spacy
import numpy as np

from auditor.perturbations import PerturbText
from auditor.evaluation.evaluate import ModelTest
from auditor.evaluation.expected_behavior import InvariantScore
from auditor.utils.dataset import get_snips_dataset
from .validation_utils import get_ner_pipeline


TEST_DATASET = [
    "please call michael",
    "please call michael bolton",
    # "how's the weather in Austin",
    # "Alexa, set timer for 5 minutes",
]

def model_predict(input_text):
    # print(f'Input text: {input_text}')
    out_len = 2
    inp_len = len(input_text)
    return np.ones((inp_len, out_len)) 


class TestSNIPS(unittest.TestCase):
    def setUp(self) -> None:
        ner_pipeline = get_ner_pipeline()
        self.snips_split = 'test'
        self.snips_data = get_snips_dataset(self.snips_split, num_samples=100)
        print(f'Perturbing {len(self.snips_data)} samples from snips {self.snips_split}-set')
        self.perturber = PerturbText(
            self.snips_data,
            ner_pipeline=ner_pipeline,
            batch_size=8,
            perturbations_per_sample=5,
        )
        return

    def test_model_eval(self):
        perturbed_dataset = self.perturber.perturb_names()
        print(f'Pertubed {len(perturbed_dataset.data)} sentences out of {perturbed_dataset.original_dataset_size} snips sentences')
        print(perturbed_dataset.data)
        invariant_behavior = InvariantScore()
        model_eval = ModelTest(
            perturbed_dataset=perturbed_dataset,
            expected_behavior=invariant_behavior,
        )
        test_details = model_eval.evaluate(model_predict=model_predict)
        print(test_details)
        return