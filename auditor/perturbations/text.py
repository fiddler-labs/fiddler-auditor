"""
Perturbations for text inputs
"""
import logging
import enum
from typing import List, Optional, Callable
from tqdm import tqdm
import re

from checklist.perturb import Perturb, process_ret
import numpy as np

from auditor.utils.data import batchify, PerturbedTextDataset
from auditor.utils.logging import get_logger
from auditor.generations.paraphrase import generate_similar_sentences
from auditor.perturbations.constants import OPENAI_CHAT_COMPLETION
from auditor.utils.similarity import (
    DEFAULT_SENTENCE_XFMR,
    load_similarity_model,
    compute_similarity
)
from auditor.utils.misc import simulate_typos

COUNTRIES = [x.lower() for x in Perturb.data['country']]
CITIES = [x.lower() for x in Perturb.data['city']]

LOG = get_logger(__name__)


@enum.unique
class PerturbationType(enum.Enum):
    """String representation for perturbations"""
    perturb_names = 'Names'
    perturb_location = 'Locations'
    perturb_number = 'Numbers'
    pertrub_typos = 'Typos'
    paraphrase = 'Paraphrase'

    def __str__(self) -> str:
        return str(self.value)


def custom_perturb_location(
    doc,
    meta=True,
    seed=None,
    n=10,
):
    if seed is not None:
        np.random.seed(seed)
    ents = [x.text for x in doc.ents if np.all([a.ent_type_ == 'GPE' for a in x])]  # noqa: E501
    ents = []
    for x in doc.ents:
        if np.all([a.ent_type_ == 'GPE' for a in x]):
            ents.append(x.text)
    ret = []
    metadata = []
    for x in ents:
        if x.lower() in COUNTRIES:
            names = Perturb.data['country'][:50]
        else:
            if x.lower() not in CITIES:
                LOG.debug(
                    f'Location {x} not found in city or country. '
                    'Assuming city.'
                )
            names = Perturb.data['city'][:100]
        sub_re = re.compile(r'\b%s\b' % re.escape(x))
        to_use = np.random.choice(names, n)
        ret.extend([sub_re.sub(n, doc.text) for n in to_use])
        metadata.extend([(x, n) for n in to_use])
    return process_ret(ret=ret, ret_m=metadata, meta=True, n=n)


def create_perturbed_dataclass(
    perturbed_ds,
    perturbations_per_sample: int,
    original_dataset_size: int,
    perturbation_type: str
):
    total_perturbations = (
        len(perturbed_ds.data) * perturbations_per_sample
    )
    return PerturbedTextDataset(
        data=perturbed_ds.data,
        metadata=perturbed_ds.meta,
        total_perturbations=total_perturbations,
        original_dataset_size=original_dataset_size,
        perturbations_per_sample=perturbations_per_sample,
        perturbation_type=perturbation_type,
    )


class PerturbText:
    """
    Class to perturb text inputs.
    """
    def __init__(
        self,
        data: List[str],
        ner_pipeline: Callable,
        batch_size: int = 8,
        perturbations_per_sample: int = 1,
        seed: int = 42,
    ):
        """Initialization function

        Args:
            data (List[str]): data to be perturbed in form on a list
            ner_pipeline (Callable):
                A function that returns entities in spacy compatible format.
                ``Example: ner_pipeline = spacy.load("en_core_web_trf").pipe``
            batch_size (int, optional):
                Batch size to be used for the NER pipeline.
                Defaults to 8.
            perturbations_per_sample (int, optional):
                Number of perturbations to be generated per datapoint.
                Defaults to 1.
            seed (int, optional): Seed.
                Defaults to 42.
        """
        self.data = data
        self.ner_pipeline = ner_pipeline
        self.batch_size = batch_size
        self.perturbations_per_sample = perturbations_per_sample
        self.seed = seed
        self._data_parsed = None
        return

    def _run_entity_extraction(self):
        self._data_parsed = []
        LOG.info("Parsing the dataset to extract entities")
        for batch_data in tqdm(
            batchify(self.data, batch_size=self.batch_size),
            total=len(self.data)//self.batch_size
        ):
            self._data_parsed.extend(
                self.ner_pipeline(batch_data)
            )
        return

    def perturb_names(
            self,
            first_only: bool = False,
            last_only: bool = False,
            metadata: bool = True,
    ) -> PerturbedTextDataset:
        """Perturb names of the dataset

        Args:
            first_only (bool, optional): Perturb first names only.
                Defaults to False.
            last_only (bool, optional): Perturb last names only.
                Defaults to False.
            metadata (bool, optional): Return metadata. Defaults to True.

        Returns:
            PerturbedTextDataset: Perturbed dataset object
        """
        LOG.info("Perturbing names of the dataset.")
        if self._data_parsed is None:
            self._run_entity_extraction()
        checklist_data = Perturb.perturb(
            self._data_parsed,
            perturb_fn=Perturb.change_names,
            keep_original=True,
            meta=metadata,
            n=self.perturbations_per_sample,
            first_only=first_only,
            last_only=last_only,
            seed=self.seed,
        )
        LOG.info(
            (
                f'Perturbed names in {len(checklist_data.data)} out of '
                f'{len(self._data_parsed)} sentences in the dataset'
            )
        )
        return create_perturbed_dataclass(
            checklist_data,
            perturbations_per_sample=self.perturbations_per_sample,
            original_dataset_size=len(self.data),
            perturbation_type=PerturbationType.perturb_names,
        )

    def perturb_location(
            self,
            metadata: bool = True,
    ) -> PerturbedTextDataset:
        """Perturb location entities in the dataset

        Args:
            metadata (bool, optional): Return metadata. Defaults to True.

        Returns:
            PerturbedTextDataset: Perturbed dataset object
        """
        LOG.info("Perturbing locations of the dataset")
        if self._data_parsed is None:
            self._run_entity_extraction()
        checklist_data = Perturb.perturb(
            self._data_parsed,
            perturb_fn=custom_perturb_location,
            keep_original=True,
            meta=metadata,
            n=self.perturbations_per_sample,
            seed=self.seed,
        )
        LOG.info(
            (
                f'Perturbed locations in {len(checklist_data.data)} out of '
                f'{len(self._data_parsed)} sentences in the dataset'
            )
        )
        return create_perturbed_dataclass(
            checklist_data,
            perturbations_per_sample=self.perturbations_per_sample,
            original_dataset_size=len(self.data),
            perturbation_type=PerturbationType.perturb_location,
        )

    def perturb_number(
        self,
        metadata: bool = True,
    ) -> PerturbedTextDataset:
        """Perturb numerical entities in the dataset

        Args:
            metadata (bool, optional): Return metadata. Defaults to True.

        Returns:
            PerturbedTextDataset: Perturbed dataset object
        """
        LOG.info("Perturbing numerical quantities of the dataset")
        if self._data_parsed is None:
            self._run_entity_extraction()
        checklist_data = Perturb.perturb(
            self._data_parsed,
            perturb_fn=Perturb.change_number,
            keep_original=True,
            meta=metadata,
            n=self.perturbations_per_sample,
            seed=self.seed,
        )
        LOG.info(
            (
                'Perturbed numerical quantities in '
                f'{len(checklist_data.data)} out of '
                f'{len(self._data_parsed)} sentences in the dataset'
            )
        )
        return create_perturbed_dataclass(
            checklist_data,
            perturbations_per_sample=self.perturbations_per_sample,
            original_dataset_size=len(self.data),
            perturbation_type=PerturbationType.perturb_number,
        )

    def perturb_typos(
            self,
            typo_probability: float = 0.02
    ) -> PerturbedTextDataset:
        """Perturb the dataset by introducing simulated user typos

        Args:
            temprature: rate of errors

        Returns:
            PerturbedTextDataset: Perturbed dataset object
        """

        LOG.info("Perturbing user typos in the dataset.")

        perturbed_dataset = []
        total_perturbations = 0
        for sentence in self.data:
            similar_sentences = []
            for i in range(0, self.perturbations_per_sample):
                similar_sentences.append(
                    simulate_typos(
                        sentence,
                        typo_probability
                    )
                )
            perturbed_dataset.append([sentence] + similar_sentences)
            total_perturbations += len(similar_sentences)

        return PerturbedTextDataset(
            data=perturbed_dataset,
            metadata=None,
            total_perturbations=total_perturbations,
            original_dataset_size=len(self.data),
            perturbations_per_sample=self.perturbations_per_sample,
            perturbation_type=PerturbationType.pertrub_typos,
        )

    def paraphrase(
        self,
        model: Optional[str] = OPENAI_CHAT_COMPLETION,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        similarity_model: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> PerturbedTextDataset:
        """Perturb the sentence by paraphrasing.

        Args:
            model (str, optional): Model to use for paraphrasing.
                Defaults to ''gpt-3.5-turbo'.
            temperature (float, optional): Tempertaure for generations.
                Defaults to 0.0.
            api_key (str) : openai API key
            similarity_model : Model to use for scoring the similarity of
                perturbations.
            api_version (str, optional): openai API version

        Returns:
            PerturbedTextDataset: Perturbed dataset object
        """
        perturbed_dataset = []
        total_perturbations = 0
        if similarity_model is None:
            similarity_model = DEFAULT_SENTENCE_XFMR
        logging.info(
            f'Loading the sentence similarity model: {similarity_model}'
        )
        similarity_model = load_similarity_model(similarity_model)
        dataset_metadata = []
        for sentence in self.data:
            similar_sentences = generate_similar_sentences(
                sentence=sentence,
                api_key=api_key,
                model=model,
                api_version=api_version,
                num_sentences=self.perturbations_per_sample,
                temperature=temperature,
            )
            perturbed_dataset.append([sentence] + similar_sentences)
            total_perturbations += len(similar_sentences)
            metadata = [None]
            for new_sent in similar_sentences:
                sim_score = compute_similarity(
                    similarity_model,
                    sentence,
                    new_sent,
                )
                metadata.append([sim_score])
            dataset_metadata.append(metadata)

        return PerturbedTextDataset(
            data=perturbed_dataset,
            metadata=dataset_metadata,
            total_perturbations=total_perturbations,
            original_dataset_size=len(self.data),
            perturbations_per_sample=self.perturbations_per_sample,
            perturbation_type=PerturbationType.paraphrase,
        )
