# Shree KRISHNAya Namaha
# Similar to MSE01, but uses disocc mask to estimate error only in disocc region; for SceneNet

import numpy
import torch
from pathlib import Path
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class MSE(LossFunctionParent):
    def compute_loss(self, input_dict: dict, output_dict: dict):
        disocc_mask4 = input_dict['disocclusion_mask4'] / 255
        frame4 = input_dict['frame4'] * disocc_mask4
        pred_frame4 = output_dict['predicted_frame4'] * disocc_mask4
        mse = torch.nn.MSELoss()(frame4, pred_frame4)
        return mse
