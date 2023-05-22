from typing import Any, List, Optional, Dict, Literal
from dataclasses import dataclass
import enum
import os

import numpy as np
import pandas as pd


RENDER_STR = {
    'perturbed_prompts': 'Perturbed Prompts',
    'perturbed_generations': 'Generations',
    'similarity_generations': 'Similarity [Generations]',
    'test_result': 'Result'
}


def batchify(data, batch_size=8):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


@enum.unique
class LLMEvalType(enum.Enum):
    """String representation for evaluation type"""
    robustness = 'Robustness'
    correctness = 'Correctness'

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class TestResult:
    input: Any
    output: np.array
    result: int
    test_metric: Any
    original_input: Any = None
    metadata: Any = None

    def __str__(self):
        return (
            f'Perturbed Input: {self.input}, '
            f'Original Input: {self.original_input}, '
            f'Output: {self.output}, '
            f'Result: {self.result}, '
            f'test_metric: {self.test_metric}, '
        )


class TestSummary:
    def __init__(
        self,
        results: List[List[TestResult]],
        robust_accuracy: float,
        description: str,
        total_perturbations: int,
        original_dataset_size: int,
        perturbations_per_sample: int,
        perturbation_type: str,
    ) -> None:
        self._index = -1
        self.description = description
        self.robust_accuracy = robust_accuracy
        self.results = results
        self.total_perturbations = total_perturbations
        self.original_dataset_size = original_dataset_size
        self.perturbations_per_sample = perturbations_per_sample
        self.perturbation_type = perturbation_type
        return

    def __len__(self):
        perturbations = 0
        for pset in self.results:
            for r in pset:
                perturbations += 1
        return perturbations

    def __str__(self):
        ret_list = [
            f'Description: {self.description}',
            f'Perturbation Type: {self.perturbation_type}',
            f'Robust accuracy: {self.robust_accuracy}',
            f'Total perturbations: {self.total_perturbations}',
        ]
        for perturbation_results in self.results:
            for p in perturbation_results:
                ret_list.append(str(p))
        return '\n'.join(ret_list)

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        try:
            return self.results[self._index]
        except IndexError:
            self._index = -1
            raise StopIteration


class TestSuiteSummary:
    def __init__(
        self,
        description: str,
    ) -> None:
        self.description = description
        self.summaries = []
        self._index = -1
        return

    def __len__(self):
        return len(self.summaries)

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        try:
            return self.summaries[self._index]
        except IndexError:
            self._index = -1
            raise StopIteration

    def add(
        self,
        test_summary: TestSummary,
    ) -> None:
        if not isinstance(test_summary, TestSummary):
            raise ValueError('Expected summary of type TestSummary')
        self.summaries.append(
            test_summary
        )
        return

    def __str__(self) -> str:
        suite_summary_str = ['Testsuite Summary:\n']
        for summary in self.summaries:
            suite_summary_str.append(
                str(summary)
            )
        return '\n'.join(suite_summary_str)


@dataclass
class PerturbedTextDataset:
    data: List[List[str]]
    metadata: Any
    total_perturbations: int
    original_dataset_size: int
    perturbations_per_sample: int
    perturbation_type: str


class LLMEvalResult:
    """Class to store LLM evaluation results"""
    def __init__(
        self,
        original_prompt: str,
        pre_context: str,
        post_context: str,
        perturbed_prompts: List[str],
        perturbed_generations: List[str],
        result: List[int],
        metric: List[Dict[str, Any]],
        expected_behavior_desc: str,
        evaluation_type: Literal[LLMEvalType.robustness, LLMEvalType.correctness],  # noqa: E501
        reference_generation: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, str]] = None,
    ) -> None:
        self.original_prompt = original_prompt
        self.pre_context = pre_context
        self.post_context = post_context
        self.perturbed_prompts = perturbed_prompts
        self.perturbed_generations = perturbed_generations
        self.reference_generation = reference_generation
        self.generation_kwargs = generation_kwargs
        self.result = result
        self.metric = metric
        self.expected_behavior_desc = expected_behavior_desc
        self.evaluation_type = evaluation_type

    def _repr_html_(self):
        pd.options.display.max_colwidth = 500
        return LLMEvalResult.render_llm_results(
            original_prompt=self.original_prompt,
            pre_context=self.pre_context,
            post_context=self.post_context,
            reference_generation=self.reference_generation,
            evaluation_type=self.evaluation_type.value,
            perturbed_generations=self.perturbed_generations,
            perturbed_prompts=self.perturbed_prompts,
            test_result=self.result,
            metric=self.metric,
            generation_kwargs=self.generation_kwargs,
            expected_behavior_desc=self.expected_behavior_desc,
        )

    def __repr__(self) -> str:
        return LLMEvalResult.render_llm_results(
            original_prompt=self.original_prompt,
            pre_context=self.pre_context,
            post_context=self.post_context,
            reference_generation=self.reference_generation,
            evaluation_type=self.evaluation_type.value,
            perturbed_generations=self.perturbed_generations,
            perturbed_prompts=self.perturbed_prompts,
            test_result=self.result,
            metric=self.metric,
            generation_kwargs=self.generation_kwargs,
            expected_behavior_desc=self.expected_behavior_desc,
        )

    @staticmethod
    def render_llm_results(
        original_prompt: str,
        pre_context: str,
        post_context: str,
        reference_generation: str,
        evaluation_type: str,
        perturbed_prompts: List[str],
        perturbed_generations: List[str],
        test_result: List[int],
        metric: List[Dict[str, Any]],
        generation_kwargs: Optional[Dict[str, str]] = None,
        expected_behavior_desc: Optional[str] = None,
    ) -> str:
        # report type
        report_type = f'{evaluation_type} report'
        # metric name
        metric_name = list(metric[0].keys())[0]
        metric_values = [i[metric_name] for i in metric]
        render_dict = {
            RENDER_STR['perturbed_prompts']: perturbed_prompts,
            RENDER_STR['perturbed_generations']: perturbed_generations,
            metric_name: metric_values,
            RENDER_STR['test_result']: test_result,
        }
        render_df = pd.DataFrame.from_dict(render_dict)
        render_df.sort_values(
            by=metric_name,
            inplace=True,
            ascending=True,
            ignore_index=True,
        )
        styled_df = LLMEvalResult._format_df(render_df)
        # generation Details:
        if generation_kwargs is not None:
            generation_details = ''
            for k, v in generation_kwargs.items():
                generation_details += f'<b>{k}:</b> {v}\n'
        else:
            generation_details = 'Sorry, generation details unavailable.'

        if pre_context is not None:
            pre_info = f'<b>Pre Context:</b> {pre_context}\n'
        else:
            pre_info = ''

        prompt_info = (
            f'<b>Prompt:</b> {original_prompt}\n'
        )
        if post_context is not None:
            post_info = f'<b>Post Context:</b> {post_context}\n'
        else:
            post_info = ''
        # generation prefix
        if evaluation_type == LLMEvalType.robustness.value:
            gen_prefix = ''
        else:
            gen_prefix = 'Reference '
        ref_gen = (
            f'<b>{gen_prefix}Generation:</b> {reference_generation}'
        )
        robustness_info = ''
        if expected_behavior_desc is not None:
            robustness_info += (
                '<b>Desired behavior:</b> '
                f'{expected_behavior_desc}\n'
            )
        passed_tests = int(sum(test_result))
        total_tests = len(test_result)
        robustness_info += f'<b>Summary: {passed_tests}/{total_tests} passed.</b>\n'
        return (
            f'<div style="border: thin solid rgb(41, 57, 141); padding: 10px;">'
            f'<h3 style="text-align: center; margin: auto;">Prompt Evaluation\n</h3><hr><pre>'
            f'<h4 style="text-align: center; margin: auto;">Generation Details\n</h4>'
            f'{generation_details}'
            f'<hr><h4 style="text-align: center; margin: auto;">Prompt Details\n</h4>'
            f'{pre_info}'
            f'{prompt_info}'
            f'{post_info}'
            f'{ref_gen}'
            f'<hr><h4 style="text-align: center; margin: auto;">{report_type}\n</h4>'
            # f'{render_df._repr_html_()}'
            f'{robustness_info}'
            f'{styled_df.to_html()}'
            f'</div>'
        )

    def save(self, file_path: str) -> None:
        """Save results in HTML format.

        Args:
            file_path (str): Path to store results in HTML format
        """
        if os.path.exists(file_path):
            raise FileExistsError(f'File: {file_path} already exists.')
        try:
            with open(file_path, 'w') as fid:
                fid.write(self._repr_html_())
        except FileNotFoundError:
            print(f'Unable to write to file path: {file_path}')
            raise

    @staticmethod
    def _format_df(df):
        def highlight_rows(row):
            value = row.loc[RENDER_STR['test_result']]
            if value > 0:
                color = '#77BBFF'  # blue
            else:
                color = '#FD9275'  # red
            return ['background-color: {}'.format(color) for r in row]
        styled_df = df.style.apply(highlight_rows, axis=1)
        return styled_df.format(precision=2)
