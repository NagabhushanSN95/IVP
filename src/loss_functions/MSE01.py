# Shree KRISHNAya Namaha
# MSE loss in infilled regions; for VEED

import numpy
import torch
from pathlib import Path
from loss_functions.LossFunctionParent01 import LossFunctionParent


def get_mse_loss_function():
    loss_function = torch.nn.MSELoss()
    return loss_function


class MSE(LossFunctionParent):
    def compute_loss(self, input_dict: dict, output_dict: dict):
        true_frame2 = input_dict['frame4']
        pred_frame2 = output_dict['predicted_frame4']
        mse = torch.nn.MSELoss()(true_frame2, pred_frame2)
        return mse
