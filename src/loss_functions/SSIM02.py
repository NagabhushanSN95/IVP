# Shree KRISHNAya Namaha
# Similar to MSE02.py, to compute loss only in disocc region; for SceneNet

import numpy
import torch
from math import exp
from pathlib import Path
import torch.nn.functional as F
from torch.autograd import Variable
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class SSIM(LossFunctionParent):
    def __init__(self):
        self.window_size = 11
        self.num_channels = 3
        self.window = self.create_window(self.window_size, self.num_channels)
        return

    def compute_loss(self, input_dict: dict, output_dict: dict):
        disocc_mask4 = input_dict['disocclusion_mask4'] / 255
        frame4 = input_dict['frame4']
        pred_frame4 = output_dict['predicted_frame4']
        self.window = self.window.to(frame4)
        ssim = self.compute_ssim(frame4, pred_frame4, disocc_mask4, self.window, self.window_size, self.num_channels)
        return 1 - ssim

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @staticmethod
    def create_window(window_size, channel):
        _1D_window = SSIM.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def compute_ssim(img1, img2, d_mask, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)
        sigma1 = torch.sqrt(sigma1_sq + 1e-12)
        sigma2 = torch.sqrt(sigma2_sq + 1e-12)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1 * sigma2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_map *= d_mask

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


def demo1():
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
