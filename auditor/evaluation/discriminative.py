from typing import Union, Callable

import numpy as np

from auditor.utils.data import (
    batchify,
    TestResult,
    TestSummary,
    TestSuiteSummary,
    PerturbedTextDataset,
)
from auditor.evaluation.expected_behavior import (
    InvariantPrediction,
    InvariantScore,
)
from auditor.reporting import generate_robustness_report
from auditor.utils.logging import get_logger

LOG = get_logger(__name__)


class ModelTest:
    """Class to evaluate model against perturbed dataset
    """
    def __init__(
        self,
        perturbed_dataset: PerturbedTextDataset,
        expected_behavior: Union[InvariantScore, InvariantPrediction],
    ) -> None:
        """Initialization function

        Args:
            perturbed_dataset (PerturbedTextDataset): Perturbed dataset object
            expected_behavior (
                Union[InvariantScore, InvariantPrediction]
            ): Expected model behavior to test against
        """
        self._check_inputs(perturbed_dataset)
        self.perturbed_dataset = perturbed_dataset
        self.expected_behavior = expected_behavior
        self._test_completed = False
        return

    @staticmethod
    def _check_inputs(perturbed_dataset):
        LOG.debug(
            'Conducting sanity check if the perturbed data '
            'is compatibe with model'
        )
        if len(perturbed_dataset.data) < 1:
            raise ValueError('Recieved empty perturbed data.')

    def _sanity_check(
        self,
        model_predict,
    ):
        # verify set-up
        try:
            test_input = [self.perturbed_dataset.data[0][0]]
            _ = self.run_model(model_predict, test_input)
        except Exception as e:
            LOG.error('Unable to run the model with the perturbed data')
            print(self.perturbed_dataset)
            raise e
        LOG.info(
            'Sanity check for perturbed inputs and model prediction '
            'method - passed'
        )
        return

    def evaluate(
        self,
        model_predict: Callable,
    ) -> TestSummary:
        """Evaluate the model against perturbed dataset and expected behavior

        Args:
            model_predict (Callable):
                A callable model prediction function than can consume
                perturbed data and return model scores.

        Returns:
            TestSummary: Test summary object
        """
        self._sanity_check(model_predict)
        perturbed_output = []
        # TODO: Should test continue when batch prediction fails?
        test_results = []
        LOG.info(
            f'Started model evaluation with perturbation type '
            f'{self.perturbed_dataset.perturbation_type}'
        )
        for perturbed_samples, metadata_samples in zip(
            self.perturbed_dataset.data, self.perturbed_dataset.metadata
        ):
            perturbed_output.append([])
            # step 1: get predictions
            for batch_samples in batchify(perturbed_samples, batch_size=1):
                try:
                    batch_output = self.run_model(model_predict, batch_samples)
                    for single_output in batch_output:
                        perturbed_output[-1].append(np.array(single_output))
                except Exception as e:
                    LOG.error(
                        'Model forward inference failed for the inputs '
                        f'{batch_samples}'
                    )
                    raise e
            # step 2: check the predictions
            test_results.append([])
            test_check = self.expected_behavior.check(perturbed_output[-1])
            # step 3: write results
            for inp, out, mdata, (test_status, metric) in zip(
                perturbed_samples,
                perturbed_output[-1],
                metadata_samples,
                test_check
            ):
                test_results[-1].append(
                    TestResult(
                        input=inp,
                        output=out,
                        result=test_status,
                        test_metric=metric,
                        original_input=perturbed_samples[0],
                        metadata=mdata,
                    )
                )
        robust_accuracy = self.compute_accuracy(test_results)
        LOG.info(f'Robust Accuracy: {robust_accuracy*100.}')
        LOG.info(
            'Completed model evaluation with perturbation type '
            f'{self.perturbed_dataset.perturbation_type}'
        )
        self.test_results = TestSummary(
            results=test_results,
            robust_accuracy=robust_accuracy,
            description=self.expected_behavior.behavior_description(),
            total_perturbations=self.perturbed_dataset.total_perturbations,
            original_dataset_size=self.perturbed_dataset.original_dataset_size,
            perturbations_per_sample=self.perturbed_dataset.perturbations_per_sample,  # noqa: E501
            perturbation_type=self.perturbed_dataset.perturbation_type.value,
        )
        # set the test completed to true
        self._test_completed = True
        return self.test_results

    def run_model(self, model_predict, input_data):
        model_output = model_predict(input_data)
        if len(model_output) != len(input_data):
            raise ValueError(
                'Mismatch in no. of inputs '
                f'({len(input_data)}) and outputs ({len(model_output)})'
            )
        return model_output

    def compute_accuracy(self, test_results):
        correct = 0
        total_perturbations = 0
        for perturbation_set in test_results:
            for result in perturbation_set:
                total_perturbations += 1
                if result.result > 0:
                    correct += 1
        robust_accuracy = float(correct)/total_perturbations
        return robust_accuracy


class TestSuite:
    """Enables testing model against several perturbations
        and expected behavior
    """
    def __init__(
        self,
        model_predict: Callable,
        description: str,
    ) -> None:
        """Init function

        Args:
            model_predict (Callable): A callable model prediction function
                than can consume perturbed data and return model scores.
            description (str): Test suite description to be used while
            generating reports.
        """
        self.model_predict = model_predict
        self.description = description
        self.tests = []
        return

    def add(
        self,
        test: ModelTest,
    ) -> None:
        """Method to add tests to the TestSuite

        Args:
            test (ModelTest): model test
        """
        test._sanity_check(self.model_predict)
        self.tests.append(test)
        return

    def evaluate(
        self,
    ) -> TestSuiteSummary:
        """Evaluate the tests in the TestSuite

        Returns:
            TestSuiteSummary: Returns TestSuiteSummary object
        """
        self.suite_results = TestSuiteSummary(
            description=self.description,
        )
        LOG.info(f'Evaluating test suite with {len(self.tests)} tests.')
        for t in self.tests:
            t.evaluate(self.model_predict)
            self.suite_results.add(t.test_results)
        return self.suite_results

    @staticmethod
    def generate_html_report(
        suite_summary: TestSuiteSummary,
        model_name: str,
        save_dir: str,
    ) -> None:
        """Generate HTML robustness report

        Args:
            suite_summary (TestSuiteSummary): Summary data object returned by
                the evaluate method
            model_name (str): Model name to be used during report generation
            save_dir (str): Directory to save the report.
        """
        generate_robustness_report(
            suite_summary=suite_summary,
            model_name=model_name,
            save_dir=save_dir,
            logger=LOG,
        )
        return
