# Shree KRISHNAya Namaha
# A Factory method that returns a Data Loader

import time
import datetime
import numpy

from pathlib import Path
from tqdm import tqdm


def get_data_loader(name: str, data_dirpath: Path, split_name: str, patch_size=None):
    if name == 'Veed':
        from data_loaders.Veed import OurDataLoader
        data_loader = OurDataLoader(data_dirpath, split_name, patch_size)
    elif name == 'SceneNet':
        from data_loaders.SceneNet import OurDataLoader
        data_loader = OurDataLoader(data_dirpath, split_name, patch_size)
    else:
        raise RuntimeError(f'Unknown data loader: {name}')
    return data_loader
