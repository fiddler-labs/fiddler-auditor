from pathlib import Path
import yaml

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

def get_test_data(test_file):
    test_yml = Path(str(Path(test_file).name).replace('.py', '.yml'))
    file_path = Path(__file__).parent / Path('test_data') / test_yml
    with open(file_path, 'r') as fid:
        test_data = yaml.safe_load(fid)
    return test_data
