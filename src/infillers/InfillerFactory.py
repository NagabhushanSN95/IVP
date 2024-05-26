# Shree KRISHNAya Namaha
# A Factory method that returns an Infiller

import numpy
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot


def get_infiller(name: str, read_off_estimator=None):
    if name == 'Infiller01':
        from infillers.Infiller01 import Infiller
        model = Infiller(read_off_estimator)
    else:
        raise RuntimeError(f'Unknown infiller network: {name}')
    return model
