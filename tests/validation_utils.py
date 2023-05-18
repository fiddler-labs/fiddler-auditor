import spacy
from spacy.cli import download


def get_ner_pipeline(model="en_core_web_trf"):
    try:
        pipe = spacy.load(model).pipe
    except OSError:
        download(model)
        pipe = spacy.load(model).pipe
    finally:
        return pipe
