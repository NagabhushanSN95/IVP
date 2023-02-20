# Shree KRISHNAya Namaha
# Utility functions for Data generation (IV prior estimation)

import time
import numpy
import pandas
import datetime
import traceback
from pathlib import Path
from typing import Optional, Tuple, List

import Imath
import OpenEXR
import skimage.io
from tqdm import tqdm

from .Warp_utils import Warper

class InfillingVectorPrior:
    def __init__(self, resolution: tuple, threshold: int = 50):
        self.h, self.w = resolution
        self.c = 2
        self.threshold = threshold
        self.window_size = 11
        self.a = self.window_size // 2
        self.save_est_IV = False    # set to true if the estimated IV also need to be saved
        print("\n 'save_est_IV' set to False; Only the Warped infilling vectors will be saved")
        time.sleep(1)
        self.grid = Warper.create_grid(self.h, self.w)

        return

    def estimate_infilling_vectors(self, frame3: numpy.ndarray, mask3: numpy.ndarray):
        infilling_vector_map = numpy.zeros(shape=(self.h, self.w, self.c), dtype=int)
        missing_pixels = numpy.argwhere(~mask3)
        for mp in missing_pixels:
            y, x = mp
            frame2_patch = frame3[y - 2:y + 3, x - 2:x + 3]

            if frame2_patch[:, :, 0].size != 25:
                continue

            candidate_locations = []
            candidate_frame2_patches = []

            # Check if there is a left valid patch
            known_pixels = numpy.argwhere(mask3[y, :x])
            if len(known_pixels) > 1:
                xc = known_pixels.max() - 1
                candidate_locations.append((y, xc))
                self.add_patch_around(candidate_frame2_patches, frame3, y, xc)

            # Check if there is a right valid patch
            known_pixels = numpy.argwhere(mask3[y, x:])
            if len(known_pixels) > 1:
                xc = x + known_pixels.min() + 1
                candidate_locations.append((y, xc))
                self.add_patch_around(candidate_frame2_patches, frame3, y, xc)

            # Check if there is a top valid patch
            known_pixels = numpy.argwhere(mask3[:y, x])
            if len(known_pixels) > 1:
                yc = known_pixels.max() - 1
                candidate_locations.append((yc, x))
                self.add_patch_around(candidate_frame2_patches, frame3, yc, x)

            # Check if there is a bottom valid patch
            known_pixels = numpy.argwhere(mask3[y:, x])
            if len(known_pixels) > 1:
                yc = y + known_pixels.min() + 1
                candidate_locations.append((yc, x))
                self.add_patch_around(candidate_frame2_patches, frame3, yc, x)

            candidates_frame_rmse = self.compute_rmse(candidate_frame2_patches, frame2_patch)
            if (len(candidates_frame_rmse) > 0) and (min(candidates_frame_rmse) <= self.threshold):
                argmin_rmse = numpy.argmin(candidates_frame_rmse)
                final_location = candidate_locations[argmin_rmse]
            else:
                final_location = (y, x)

            infilling_vector_map[y, x] = (final_location[1] - x, final_location[0] - y)
        return infilling_vector_map


    def smoothen_infilling_vector(self, infilling_vector: numpy.ndarray, mask: numpy.ndarray):
        known_mask = (infilling_vector[:, :, 0] != 0) | (infilling_vector[:, :, 1] != 0)
        padded_infilling_vector = numpy.zeros(shape=(self.h + 2 * self.a, self.w + 2 * self.a, self.c), dtype=infilling_vector.dtype)
        padded_mask = numpy.zeros(shape=(self.h + 2 * self.a, self.w + 2 * self.a), dtype=known_mask.dtype)
        padded_infilling_vector[self.a:-self.a, self.a:-self.a] = infilling_vector
        padded_mask[self.a:-self.a, self.a:-self.a] = known_mask

        sum_infilling_vector = numpy.zeros(infilling_vector.shape, infilling_vector.dtype)
        sum_mask = numpy.zeros(known_mask.shape, dtype='uint8')
        for m in range(self.window_size):
            for n in range(self.window_size):
                sum_infilling_vector += padded_infilling_vector[m:m + self.h, n:n + self.w]
                sum_mask += padded_mask[m:m + self.h, n:n + self.w]
        with numpy.errstate(invalid='ignore'):
            avg_infilling_vector = sum_infilling_vector / sum_mask[:, :, None]
        avg_infilling_vector[numpy.isnan(avg_infilling_vector)] = 0
        final_infilling_vector = avg_infilling_vector * mask[:, :, None]
        return final_infilling_vector


    def warp_infilling_vector(self, infilling3: numpy.ndarray, trans_points3: numpy.ndarray):

        trans_coordinates3 = trans_points3[:, :, :2, 0] / trans_points3[:, :, 2:3, 0]
        trans_depth3 = trans_points3[:, :, 2, 0]
        flow12 = trans_coordinates3 - self.grid
                
        warped_infilling_vector, mask4 = Warper.bilinear_splatting(infilling3, None, trans_depth3, flow12, None, is_image=False)
        smoothed_infilling_vector = self.smoothen_infilling_vector(warped_infilling_vector, mask4)
        return smoothed_infilling_vector


    def estimate_and_warp_iv(self, new_paths_dict: dict, frame_path: Path, mask_path: Path, trans_points_path: Path):

        generated_data_dict = dict()

        frame3 = read_image(frame_path)
        mask3 = read_mask(mask_path)
        trans_points3 = read_npy(trans_points_path)

        generated_data_dict['estimated_iv'] = model.estimate_infilling_vectors(frame3, mask3).copy()
        generated_data_dict['warped_iv'] = model.warp_infilling_vector(generated_data_dict['estimated_iv'], trans_points3).copy()

        save_generated_data(generated_data_dict, new_paths_dict)                

        return

    @staticmethod
    def add_patch_around(patches_list: list, frame: numpy.ndarray, y, x):
        patch = frame[y - 2:y + 3, x - 2:x + 3]
        if patch.ndim == 2:
            req_patch_size = 5 * 5
        elif patch.ndim == 3:
            req_patch_size = 5 * 5 * 3
        else:
            raise RuntimeError('Incomprehensible patch dimensions')

        if patch.size == req_patch_size:
            patches_list.append(patch)
        return

    @staticmethod
    def compute_rmse(candidate_patches: List[numpy.ndarray], true_patch: numpy.ndarray, normalize=False):
        rmse_list = []
        for candidate_patch in candidate_patches:
            error = true_patch.astype('float') - candidate_patch.astype('float')
            rmse = numpy.sqrt(numpy.mean(numpy.square(error)))
            if normalize:
                rmse /= numpy.mean(numpy.abs(true_patch))
            rmse_list.append(rmse)
        return rmse_list