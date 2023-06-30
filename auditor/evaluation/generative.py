from typing import List, Optional, Literal, Dict

from langchain.llms.base import BaseLLM

from auditor.utils.data import (
    LLMEvalResult,
    LLMEvalType,
)
from auditor.evaluation.expected_behavior import (
    SimilarGeneration,
)
from auditor.utils.logging import get_logger
from auditor.perturbations import Paraphrase
from auditor.perturbations import TransformBase

LOG = get_logger(__name__)


class LLMEval:
    def __init__(
        self,
        llm:  BaseLLM,
        expected_behavior: SimilarGeneration,
        transformation: Optional[TransformBase] = None,
    ) -> None:
        """Class for evaluating Large Language Models (LLMs)

        Args:
            llm (BaseLLM): Langchain LLM Object
            expected_behavior (SimilarGeneration):
                Expected model behavior to evaluate against
            transformation (Optional[TransformBase], optional):
                Transformation to evaluate against.
                When not provided defaults to using auditor.perturbations.Paraphrase.  # noqa: E501
        """
        self.llm = llm
        self.expected_behavior = expected_behavior
        if transformation is None:
            self.transformation = Paraphrase()
        else:
            self.transformation = transformation
        return

    def _evaluate_generations(
        self,
        prompt: str,
        evaluation_type: Literal[LLMEvalType.robustness, LLMEvalType.correctness],  # noqa: E501
        pre_context: Optional[str] = None,
        post_context: Optional[str] = None,
        reference_generation: Optional[str] = None,
        prompt_perturbations: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> LLMEvalResult:
        """
        Evaluates generations to paraphrased prompt perturbations

        Args:
            prompt (str): Prompt to be perturbed
            evaluation_type (LLMEvalType): Evaluation type. Supported types -
            Robustness or Correctness.
            pre_context (Optional[str], optional):
                Context prior to prompt, will not be perturbed.
                Defaults to None.
            post_context (Optional[str], optional):
                Context post prompt, will not be perturbed.
                Defaults to None.
            reference_generation (Optional[str], optional):
                Reference generation to compare against. Defaults to None.
            prompt_perturbations (Optional[List[str]], optional):
                Alternative prompts to use. Defaults to None. When absent,
                method generates perturbations by paraphrasing the prompt.

        Returns:
            LLMEvalResult: Object wth evaluation results
        """
        # reference generation
        if reference_generation is None:
            LOG.debug(
                'Fetching reference generation for the prompt: '
                f'{prompt}'
            )
            reference_generation = self._get_generation(
                prompt,
                pre_context,
                post_context,
            )
        # create alternative prompt perturbations
        if prompt_perturbations is None:
            prompt_perturbations = self.generate_alternative_prompts(
                prompt=prompt,
                *args,
                **kwargs,
            )
        # include the original prompt when evaluating correctness
        if evaluation_type.value == LLMEvalType.correctness.value:
            evaluate_prompts = [prompt] + prompt_perturbations
        else:
            evaluate_prompts = prompt_perturbations

        # generations for each of the perturbed prompts
        alternative_generations = []
        for alt_prompt in evaluate_prompts:
            resp = self._get_generation(
                alt_prompt,
                pre_context,
                post_context,
            )
            alternative_generations.append(resp)

        # create test result
        metric = self.expected_behavior.check(
            reference_generation=reference_generation,
            perturbed_generations=alternative_generations,
        )
        return LLMEvalResult(
            original_prompt=prompt,
            pre_context=pre_context,
            post_context=post_context,
            reference_generation=reference_generation,
            perturbed_prompts=evaluate_prompts,
            perturbed_generations=alternative_generations,
            generation_kwargs=self._get_generation_details(),
            result=[m[0] for m in metric],
            metric=[m[1] for m in metric],
            expected_behavior_desc=self.expected_behavior.descriptor,
            evaluation_type=evaluation_type,
        )

    def _get_generation(
        self,
        prompt: str,
        pre_context: Optional[str],
        post_context: Optional[str],
    ) -> str:
        """Get generation from the model"""
        try:
            llm_input = self.construct_llm_input(
                prompt,
                pre_context,
                post_context
            )
            response = str(self.llm(llm_input))
        except Exception as err:
            LOG.error('Unable to fetch generations from the model.')
            raise err
        return response

    def construct_llm_input(
        self,
        prompt: str,
        pre_context: Optional[str],
        post_context: Optional[str],
        delimiter: str = " ",
    ) -> str:
        if pre_context is not None:
            full_prompt = pre_context + delimiter + prompt
        else:
            full_prompt = prompt
        if post_context is not None:
            full_prompt += delimiter + post_context
        return full_prompt

    def generate_alternative_prompts(
        self,
        prompt: str,
        *args,
        **kwargs,
    ) -> List[str]:
        """Generates perturbed prompts

        Args:
            prompt (str): Prompt to be perturbed
            Returns:
            List[str]: List of perturbed prompts.
        """
        return self.transformation.transform(
            prompt,
            *args,
            **kwargs,
        )

    def _get_generation_details(self) -> Dict[str, str]:
        """Returns generation related details"""
        details = {}
        if hasattr(self.llm, '_llm_type'):
            details['Provider'] = self.llm._llm_type
        if hasattr(self.llm, 'temperature'):
            details['Temperature'] = self.llm.temperature
        if hasattr(self.llm, 'model_name'):
            details['Model Name'] = self.llm.model_name
        return details

    def evaluate_prompt_robustness(
        self,
        prompt: str,
        perturbations_per_sample: Optional[int] = None,
        pre_context: Optional[str] = None,
        post_context: Optional[str] = None,
        prompt_perturbations: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> LLMEvalResult:
        """
        Evaluates robustness of generation to paraphrased prompt perturbations

        Args:
            prompt (str): Prompt to be perturbed
            perturbations_per_sample (int, optional):
                Deprecated. No of perturbation is now controlled by the
                Transform object.
            pre_context (Optional[str], optional):
                Context prior to prompt, will not be perturbed.
                Defaults to None.
            post_context (Optional[str], optional):
                Context post prompt, will not be perturbed.
                Defaults to None.
            prompt_perturbations (Optional[List[str]], optional):
                Prompt perturbations to use. Defaults to None. When absent,
                method generates perturbations by paraphrasing the prompt.

        Returns:
            LLMEvalResult: Object wth evaluation results
        """
        return self._evaluate_generations(
            prompt=prompt,
            evaluation_type=LLMEvalType.robustness,
            pre_context=pre_context,
            post_context=post_context,
            reference_generation=None,
            prompt_perturbations=prompt_perturbations,
            *args,
            **kwargs,
        )

    def evaluate_prompt_correctness(
        self,
        prompt: str,
        reference_generation: str,
        perturbations_per_sample: Optional[int] = None,
        pre_context: Optional[str] = None,
        post_context: Optional[str] = None,
        alternative_prompts: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> LLMEvalResult:
        """
        Evaluates robustness of generation to paraphrased prompt perturbations

        Args:
            prompt (str): Prompt to be perturbed
            reference_generation (str):
                Reference generation to compare against.
            perturbations_per_sample (int, optional):
                Deprecated. No of perturbation is now controlled by the
                Transform object.
            pre_context (Optional[str], optional):
                Context prior to prompt, will not be perturbed.
                Defaults to None.
            post_context (Optional[str], optional):
                Context post prompt, will not be perturbed.
                Defaults to None.
            alternative_prompts (Optional[List[str]], optional):
                Alternative prompts to use. Defaults to None. When provided no
                perturbations are generated.

        Returns:
            LLMEvalResult: Object wth evaluation results
        """
        return self._evaluate_generations(
            prompt=prompt,
            evaluation_type=LLMEvalType.correctness,
            pre_context=pre_context,
            post_context=post_context,
            reference_generation=reference_generation,
            prompt_perturbations=alternative_prompts,
            *args,
            **kwargs,
        )
