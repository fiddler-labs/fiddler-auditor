import unittest
from pathlib import Path

from sentence_transformers.SentenceTransformer import SentenceTransformer

from auditor.evaluation.evaluate import LLMEval
from auditor.evaluation.expected_behavior import (
    ModelGraded, SimilarGeneration, Toxicity
)
from .validation_utils import get_test_data

TEST_DATA = get_test_data(__file__)

class TestModelEval(unittest.TestCase):
    def setUp(self) -> None:
        return

    def test_model_graded(self):
        kwargs = TEST_DATA['ModelGraded']
        model_grader = ModelGraded()
        result = model_grader.check(**kwargs)
        grade = [r[0] for r in result]
        assert sum(grade)==4, 'Expected exactly 4/5 grades to be correct.'
        return

    def test_similar_generation(self):
        kwargs = TEST_DATA['SimilarGeneration']
        sent_xfmer = SentenceTransformer(
            'sentence-transformers/paraphrase-mpnet-base-v2'
        )
        similar_generation = SimilarGeneration(
            similarity_model=sent_xfmer,
            similarity_threshold=0.95,
        )
        result = similar_generation.check(**kwargs)
        grade = [r[0] for r in result]
        assert sum(grade)==1, 'Expected exactly 1/2 result to be correct.'
        return

    def test_valid_url(self):
        return
    
    def test_toxicity(self):
        kwargs = TEST_DATA['Toxicity']
        toxicity_check = Toxicity(threshold=0.6)
        result = toxicity_check.check(**kwargs)
        grade = [r[0] for r in result]
        print(result)
        assert sum(grade)==1, 'Expected exactly 1/2 result to be toxic.'
        return
        return