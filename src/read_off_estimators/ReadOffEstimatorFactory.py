# Shree KRISHNAya Namaha
# A Factory method that returns a read off estimation network

import numpy
from pathlib import Path
from tqdm import tqdm


def get_read_off_estimation_network(name: str):
    if name == 'Unet':
        from read_off_estimators.Unet import ReadOffEstimationNetwork
        model = ReadOffEstimationNetwork()
    else:
        raise RuntimeError(f'Unknown read off estimation network: {name}')
    return model
