from typing import List, Optional
import os
import re

import openai

from auditor.perturbations.base import TransformBase
from auditor.perturbations.constants import OPENAI_CHAT_COMPLETION


class Paraphrase(TransformBase):
    """Perturbation class that paraphrases by querying open-ai LLM
    """
    def __init__(
        self,
        model: Optional[str] = OPENAI_CHAT_COMPLETION,
        num_perturbations: int = 5,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        self._init_key(api_key)
        self._init_model(model, api_version)
        self.num_perturbations = num_perturbations
        self.temperature = temperature
        self.descriptor = (
            f'Paraphrases the original prompt with '
            f'an open-ai {self.model} model.'
        )
        self.paraphrase_instruction = (
            'Generate a bulleted list of {n} sentences '
            'with same meaning as \"{sentence}\"'
        )
        return

    def description(self) -> str:
        return self.descriptor

    def _init_key(self, api_key: str):
        """Initialize API key"""
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.api_key = api_key
        openai.api_key = api_key
        return

    def _init_model(
        self,
        model,
        api_version
    ):
        """Initialize model, engine and api version"""
        self.model = model
        self.api_version = api_version
        if openai.api_type == "azure":
            self.engine = model
            self.api_version = api_version
        else:
            self.engine = None
        return

    def transform(
        self,
        prompt: str,
    ) -> List[str]:
        prompt = self.paraphrase_instruction.format(
            n=self.num_perturbations,
            sentence=prompt
        )
        payload = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
            engine=self.engine,
            api_version=self.api_version,
        )
        return Paraphrase._process_similar_sentence_reponse(response)

    @staticmethod
    def _process_similar_sentence_reponse(response):
        generation = response['choices'][0]['message']['content']
        # Use a combination of lookahead and lookback
        # Expr extracts generations between the
        # bulltet '-' and newline character
        sim_sent = re.findall(
            r'(?<=\n-)(.*?)(?=\n)',
            '\n'+generation+'\n'
        )
        # removing whitespaces
        sim_sent = [s.strip() for s in sim_sent]
        return sim_sent
