# Shree KRISHNAya Namaha
# Computes all specified losses

import time
import numpy
import datetime
import traceback
from pathlib import Path
from matplotlib import pyplot
from typing import List, Tuple


class LossComputer:
    def __init__(self, loss_functions: List[Tuple[str, int]]):
        self.losses = {}
        for loss_name, loss_weight in loss_functions:
            self.losses[loss_name] = self.get_loss_object(loss_name), loss_weight
        return

    @staticmethod
    def get_loss_object(loss_name):
        if loss_name == 'MSE01':
            from loss_functions.MSE01 import MSE
            loss_obj = MSE()
        elif loss_name == 'MSE02':
            from loss_functions.MSE02 import MSE
            loss_obj = MSE()
        elif loss_name == 'SSIM02':
            from loss_functions.SSIM02 import SSIM
            loss_obj = SSIM()
        elif loss_name == 'SSIM01':
            from loss_functions.SSIM01 import SSIM
            loss_obj = SSIM()
        elif loss_name == 'SmoothnessLoss12':
            from loss_functions.SmoothnessLoss12 import SmoothnessLoss
            loss_obj = SmoothnessLoss()
        else:
            raise RuntimeError(f'Unknown Loss Function: {loss_name}')
        return loss_obj

    def compute_losses(self, input_dict, output_dict):
        loss_values = {}
        total_loss = 0
        for loss_name in self.losses.keys():
            loss_obj, loss_weight = self.losses[loss_name]
            loss_value = loss_obj.compute_loss(input_dict, output_dict)
            loss_values[loss_name] = loss_value
            total_loss += loss_weight * loss_value
        loss_values['Total Loss'] = total_loss
        return loss_values
