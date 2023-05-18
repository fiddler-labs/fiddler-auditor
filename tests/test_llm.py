import unittest
import os

from langchain.llms.fake import FakeListLLM
from sentence_transformers.SentenceTransformer import SentenceTransformer

from auditor.evaluation.evaluate import LLMEval
from auditor.evaluation.expected_behavior import SimilarGeneration

TEST_DATA = [
   { 
        "prompt": "What are the three primary colors?",
        "pre_context": None,
        "post_context": None,
        "reference_generation": "The three primary colors are red, blue, and yellow.",
        "responses": [
            "The three colors are red, green, and blue.",
            "Red, yellow, and blue.",
            "The primary colors are red, green, and blue.",
            "The three fundamental colors are red, green, and blue.",
            "The three primary colors are red, green, and blue.",
            "The three primary colors are red, green, and blue.",
        ]
    },
    { 
        "prompt": "What are the three primary colors?",
        "pre_context": "Answer the following thoughtfully: ",
        "post_context": None,
        "reference_generation": "The three primary colors are red, blue, and yellow.",
        "responses": [
            "The three colors are red, green, and blue.",
            "Red, yellow, and blue.",
            "The primary colors are red, green, and blue.",
            "The three fundamental colors are red, green, and blue.",
            "The three primary colors are red, green, and blue.",
            "The three primary colors are red, green, and blue.",
        ]
   },
]

SIMILARITY_MODEL = 'sentence-transformers/paraphrase-mpnet-base-v2'


class TestModelEval(unittest.TestCase):
    def setUp(self) -> None:
        # ner_pipeline = spacy.load("en_core_web_sm").pipe
        self.llm = FakeListLLM(
            responses=TEST_DATA[0]["responses"]
        )
        return

    def test_evaluate_prompt_robustness(self):
        similarity_model = SentenceTransformer(SIMILARITY_MODEL)
        similar_gen = SimilarGeneration(
            similarity_model=similarity_model,
        )
        llm_eval = LLMEval(
            llm=self.llm,
            expected_behavior=similar_gen
        )
        example = TEST_DATA[0]
        test_result = llm_eval.evaluate_prompt_robustness(
            prompt=example["prompt"],
            pre_context=example["pre_context"],
            post_context=example["post_context"],
        )
        print(test_result._repr_html_())
        return

    def test_evaluate_prompt_robustness_context(self):
        similarity_model = SentenceTransformer(SIMILARITY_MODEL)
        similar_gen = SimilarGeneration(
            similarity_model=similarity_model,
        )
        llm_eval = LLMEval(
            llm=self.llm,
            expected_behavior=similar_gen
        )
        example = TEST_DATA[1]
        test_result = llm_eval.evaluate_prompt_robustness(
            prompt=example["prompt"],
            pre_context=example["pre_context"],
            post_context=example["post_context"],
        )
        print(test_result._repr_html_())
        return

    def test_evaluate_prompt_correctness(self):
        similarity_model = SentenceTransformer(SIMILARITY_MODEL)
        similar_gen = SimilarGeneration(
            similarity_model=similarity_model,
        )
        llm_eval = LLMEval(
            llm=self.llm,
            expected_behavior=similar_gen
        )
        example = TEST_DATA[1]
        test_result = llm_eval.evaluate_prompt_correctness(
            prompt=example["prompt"],
            reference_generation=example["reference_generation"],
            pre_context=example["pre_context"],
            post_context=example["post_context"],
        )
        print(test_result._repr_html_())
        return

    def test_evaluate_prompt_correctness_manual(self):
        similarity_model = SentenceTransformer(SIMILARITY_MODEL)
        similar_gen = SimilarGeneration(
            similarity_model=similarity_model,
        )
        llm_eval = LLMEval(
            llm=self.llm,
            expected_behavior=similar_gen
        )
        example = TEST_DATA[1]
        # checking over-riding perturbations
        test_result = llm_eval.evaluate_prompt_correctness(
            prompt=example["prompt"],
            reference_generation=example["reference_generation"],
            pre_context=example["pre_context"],
            post_context=example["post_context"],
            alternative_prompts=[example["prompt"]]*5,
        )
        print(test_result._repr_html_())
        # test file save
        save_path = os.path.join(os.path.dirname(__file__), 'test_report.html')
        test_result.save(file_path=save_path)
        os.remove(save_path)
        return
