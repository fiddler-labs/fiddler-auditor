from typing import List
import os
import openai

from auditor.perturbations.constants import OPENAI_CHAT_COMPLETION

SIMILAR_SENTENCES_PROMPT = (
    'Generate a numbered list of {n} sentences '
    'with same meaning as \"{sentence}\"'
)


def generate_similar_sentences(
    sentence: str,
    api_key: str,
    model: str = OPENAI_CHAT_COMPLETION,
    num_sentences: int = 5,
    temperature: float = 0.0,
) -> List[str]:
    prompt = SIMILAR_SENTENCES_PROMPT.format(
        n=num_sentences,
        sentence=sentence
    )
    payload = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
      model=model,
      messages=payload,
      temperature=temperature,
    )
    return _process_similar_sentence_reponse(response)


def _process_similar_sentence_reponse(response):
    sim_sent = []
    lines = response['choices'][0]['message']['content'].split('\n')
    for ln in lines:
        r = ln.split('.')[1]
        sim_sent.append(r.strip())
    return sim_sent
