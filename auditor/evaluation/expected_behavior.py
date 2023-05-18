from abc import ABC, abstractmethod, abstractproperty
from typing import List, Tuple, Optional, Dict

import numpy as np
from sentence_transformers.SentenceTransformer import SentenceTransformer

from auditor.utils.similarity import compute_similarity
from auditor.utils.logging import get_logger

FAILED_TEST = 0
PASSED_TEST = 1
LOG = get_logger(__name__)


class AbstractBehavior(ABC):
    """Abstract class to help in creation of ExpectedBehavior classes
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def check(self) -> List[Tuple[bool, Dict[str, float]]]:
        raise NotImplementedError(
            'Derived class must implement the check method.'
        )

    @abstractproperty
    def behavior_description(self):
        pass


class InvariantScore(AbstractBehavior):
    """
    Class to verify if the model's output scores are invariant to
    perturbations.
    """
    def __init__(
        self,
        rel_tol: float = 0.01,
    ) -> None:
        self.rel_tol = rel_tol
        self.descriptor = (
            'Model scores are invariant '
            'to input perturbations '
            f'within a relative tolerance of {self.rel_tol*100} %'
        )
        return

    def check(
        self,
        perturbed_scores: List[np.array],
        original_score: Optional[np.array] = None,
    ) -> List[Tuple[bool, float]]:
        test_results = []
        if original_score is None:
            original_score = perturbed_scores[0]
        for s in perturbed_scores:
            try:
                rel_err = np.mean(np.abs((original_score - s)/original_score))
            except Exception:
                LOG.warn(
                    'Ran into an error while computing relative-error, '
                    'marking the test as failed'
                )
                test_results.append((FAILED_TEST,  float('nan')))
            else:
                if rel_err <= self.rel_tol:
                    test_status = PASSED_TEST
                else:
                    test_status = FAILED_TEST
                test_results.append((test_status,  rel_err))
        return test_results

    def behavior_description(self):
        return self.descriptor


class InvariantPrediction(AbstractBehavior):
    """
    Class to verify if the model's output scores are invariant to
    perturbations.
    """
    def __init__(
        self,
        top_k: int = 1,
    ) -> None:
        self.top_k = top_k
        self.descriptor = (
            f'Model\'s top-{top_k} '
            f'predictions are invariant to input perturbations.'
        )
        return

    def check(
        self,
        perturbed_scores: List[np.array],
        reference_score: Optional[np.array] = None,
    ) -> List[Tuple[bool, float]]:
        """Check if the model's predictions are invariant compared to
          reference_score.

        Args:
            perturbed_scores (List[np.array]):
                Array of model's predictions for perturbed inputs
            reference_score (Optional[np.array], optional):
                Reference model prediction to check against.
                When not provided the first entry in perturbed scores
                is used as reference score.

        Returns:
            List[Tuple[bool, float]]: _description_
        """
        test_results = []
        if reference_score is None:
            reference_score = perturbed_scores[0]
        for s in perturbed_scores:
            try:
                order_match = bool(
                    np.all(
                        np.flip(np.argsort(reference_score))[:self.top_k] ==
                        np.flip(np.argsort(s))[:self.top_k]
                    )
                )
            except Exception:
                LOG.warn(
                    'Ran into an error while matching predictions, marking '
                    'the test as failed'
                )
                test_results.append((FAILED_TEST,  float('nan')))
            else:
                test_status = PASSED_TEST if order_match else FAILED_TEST
                test_results.append((test_status,  float('nan')))
        return test_results

    def behavior_description(self):
        return self.descriptor


class SimilarGeneration(AbstractBehavior):
    """
    Class to verify if the model's generations are robust to
    perturbations.
    """
    def __init__(
        self,
        similarity_model: SentenceTransformer,
        similarity_threshold: float = 0.8,
        similarity_metric_key: str = 'Similarity [Generations]'
    ) -> None:
        self.similarity_model = similarity_model
        self.similarity_threshold = similarity_threshold
        self.similarity_metric_key = similarity_metric_key
        self.descriptor = (
            f'Model\'s generations for perturbations '
            f'are greater than {self.similarity_threshold} similarity metric '
            f'compared to the reference generation.'
        )
        return

    def check(
        self,
        perturbed_generations: List[str],
        reference_generation: str,
    ) -> List[Tuple[bool, Dict[str, float]]]:
        test_results = []
        for peturbed_gen in perturbed_generations:
            try:
                score = compute_similarity(
                    sentence_model=self.similarity_model,
                    reference_sentence=reference_generation,
                    perturbed_sentence=peturbed_gen,
                )
                if score >= self.similarity_threshold:
                    test_status = PASSED_TEST
                else:
                    test_status = FAILED_TEST
                score_dict = {
                    self.similarity_metric_key: round(score, ndigits=2)
                }
                test_results.append((test_status, score_dict))
            except Exception as e:
                LOG.error('Unable to complete semanatic similarity checks')
                raise e
        return test_results

    def behavior_description(self):
        return self.descriptor
