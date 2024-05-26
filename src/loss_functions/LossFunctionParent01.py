# Shree KRISHNAya Namaha
# Abstract parent class
# Author: Nagabhushan S N
# Date: 16/12/2020

import abc
import torch
import numpy
from tqdm import tqdm
from pathlib import Path


class LossFunctionParent:
    @abc.abstractmethod
    def compute_loss(self, input_dict: dict, output_dict: dict):
        pass
