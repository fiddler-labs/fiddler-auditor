from pathlib import Path
from typing import Optional

import pandas as pd

SNIPS_SPLITS = {
    'train': 'snips_dataset_train.csv',
    'test': 'snips_dataset_test.csv',
    'dev': 'snips_dataset_dev.csv',
}

DATA_DIR = Path(__file__).resolve().parent.parent / 'assets' / 'data'


def get_snips_dataset(
        split: str = 'test',
        num_samples: Optional[int] = None,
        dataframe: bool = False
):
    if split not in SNIPS_SPLITS:
        legit_splits = ','.join(list(SNIPS_SPLITS.keys()))
        raise ValueError(
            'Expected split type to be among '
            f'{legit_splits}, received {split}'
        )
    dataset_path = DATA_DIR / SNIPS_SPLITS[split]
    df = pd.read_csv(dataset_path)
    if num_samples is not None:
        df = df.sample(num_samples)
    if dataframe:
        return df
    else:
        return df['sentence'].values.tolist()
