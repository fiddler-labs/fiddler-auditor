import unittest

import spacy

from auditor.perturbations import PerturbText
from .validation_utils import get_ner_pipeline

from .validation_utils import get_ner_pipeline
TEST_DATASET = [
    "please call michael",
    "please call michael bolton",
    "how's the weather in Austin",
    "Set timer for 5 minutes",
]

class TestPerturbText(unittest.TestCase):
    def setUp(self) -> None:
        ner_pipeline = get_ner_pipeline()
        self.perturber = PerturbText(
            TEST_DATASET,
            ner_pipeline=ner_pipeline,
            batch_size=8,
            perturbations_per_sample=5,
        )

    def test_perturb_names(self):
        print(self.perturber.perturb_names())
    
    def test_perturb_location(self):
        print(self.perturber.perturb_location())

    def test_perturb_number(self):
        print(self.perturber.perturb_number())
    
    def test_paraphrase(self):
        similar_sentences = self.perturber.paraphrase(
            model = 'gpt-3.5-turbo',
            temperature = 0.0,
        )
        print(similar_sentences)