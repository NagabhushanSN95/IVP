# Shree KRISHNAya Namaha
# Extended from SmoothnessLoss03.py. L1 loss over flow, weighted by frame4. Smoothness is computed in unknown regions
# only.
# Author: Nagabhushan S N
# Date: 12/08/2021

import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot
import torch
import torch.nn.functional as F

from loss_functions.LossFunctionParent01 import LossFunctionParent


class SmoothnessLoss(LossFunctionParent):
    def compute_loss(self, input_dict: dict, output_dict: dict):
        frame4 = input_dict['frame4']
        mask4 = input_dict['mask4'] / 255
        iv = output_dict['read_off_values']
        image_dy, image_dx = self.compute_gradients(frame4)
        flow_dy, flow_dx = self.compute_gradients(iv)
        sm_mask_y = (mask4[:, :, 1:, :] == 0) & (mask4[:, :, :-1, :] == 0)
        sm_mask_x = (mask4[:, :, :, 1:] == 0) & (mask4[:, :, :, :-1] == 0)

        w_y = torch.exp(-10.0 * image_dy.abs().mean(dim=1).unsqueeze(1)) * sm_mask_y
        w_x = torch.exp(-10.0 * image_dx.abs().mean(dim=1).unsqueeze(1)) * sm_mask_x
        loss_value = (w_y * flow_dy.abs()).mean() + (w_x * flow_dx.abs()).mean()
        return loss_value

    @staticmethod
    def compute_gradients(image):
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        return grad_y, grad_x
