# Shree KRISHNAya Namaha
# Depth preprocessing + Unet (IVP) to estimate Infilling Vectors
# Depth preprocessor takes as input warped depth of frame n+1
# IVP takes as input warped prior and processed depth

import numpy
import torch
from pathlib import Path
import torch.nn.functional as F


class ReadOffEstimationNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_preprocessor = DepthPreprocessor()
        self.unet = Unet()
        return

    def forward(self, input_batch):
        depth_preprocessor_output = self.depth_preprocessor(input_batch)
        result_dict = depth_preprocessor_output.copy()
        unet_output = self.unet(input_batch, result_dict)
        for key in unet_output.keys():
            result_dict[key] = unet_output[key]
        return result_dict


class DepthPreprocessor(torch.nn.Module):
    def __init__(self):
        super(DepthPreprocessor, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        return

    def forward(self, input_batch):
        warped_depth = input_batch['warped_depth4']
        x1 = self.conv1(warped_depth)
        x1 = F.relu(x1)
        x2 = self.conv2(x1)
        x2 = F.relu(x2)
        x3 = self.up(x2)
        x3 = self.conv3(x3)
        x3 = F.relu(x3)
        x4 = self.up(x3)
        x4 = self.conv4(x4)
        depth_mask = F.sigmoid(x4)
        result_dict = {
            'depth_mask': depth_mask,
        }
        return result_dict


class Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = torch.nn.Conv2d(in_channels=256 + 256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=256 + 256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(in_channels=128 + 128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(in_channels=64 + 64, out_channels=2, kernel_size=3, stride=1, padding=1)
        return

    def forward(self, input_batch, previous_result):
        warped_infilling4 = input_batch['warped_infilling4']
        mask4 = input_batch['mask4'] / 255
        depth_mask = previous_result['depth_mask']
        model_input = torch.cat([warped_infilling4, depth_mask], dim=1)
        x1 = self.conv1(model_input)  # 256x256
        x1 = F.relu(x1)
        x2 = self.conv2(x1)  # 128x128
        x2 = F.relu(x2)
        x3 = self.conv3(x2)  # 64x64
        x3 = F.relu(x3)
        x4 = self.conv4(x3)  # 32x32
        x4 = F.relu(x4)
        x5 = self.conv5(x4)  # 16x16, 256
        x5 = F.relu(x5)
        x6 = self.up(x5)  # 32x32
        if x6.shape != x4.shape:
            x6 = x6[:, :, :-1]
        x6 = torch.cat([x6, x4], dim=1)  # 32x32, 256+256
        x6 = self.conv6(x6)  # 32x32, 256
        x6 = F.relu(x6)
        x7 = self.up(x6)  # 64x64
        x7 = torch.cat([x7, x3], dim=1)  # 64x64, 256+256
        x7 = self.conv7(x7)  # 64x64, 128
        x7 = F.relu(x7)
        x8 = self.up(x7)  # 128x128, 128
        x8 = torch.cat([x8, x2], dim=1)  # 128x128, 128+128
        x8 = self.conv8(x8)  # 128x128, 64
        x8 = F.relu(x8)
        x9 = self.up(x8)  # 256x256, 64
        x9 = torch.cat([x9, x1], dim=1)  # 256x256, 64+64
        infilled_infilling4 = self.conv9(x9)  # 256x256, 2
        read_off_values = infilled_infilling4 * (1 - mask4)
        result_dict = {
            'read_off_values': read_off_values,
        }
        return result_dict
