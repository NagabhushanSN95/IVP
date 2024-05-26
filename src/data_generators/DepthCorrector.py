# Shree KRISHNAya Namaha

import time
import math
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

from utils import *
from .PoseWarping import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class DepthCorrector:
    def __init__(self, intensity_threshold: int = 30, window_size: int = 11):
        self.intensity_threshold = intensity_threshold
        self.window_size = window_size
        return

    def correct_depth(self, warper: Warper, depth1: numpy.ndarray, frame1: numpy.ndarray, frame2: numpy.ndarray,
                      transformation1: numpy.ndarray, transformation2: numpy.ndarray, intrinsic: numpy.ndarray, new_paths_dict: dict):
        h, w = depth1.shape
        generated_data_dict = dict()

        depth1 = numpy.clip(depth1, a_min=0, a_max=1e10)
        trans_points1 = warper.compute_transformed_points(depth1, None, transformation1, transformation2, intrinsic, None)
        trans_coordinates = trans_points1[:, :, :2, 0] / trans_points1[:, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, 2, 0]

        grid = warper.create_grid(h, w)
        flow12 = trans_coordinates - grid

        warped_frame2, mask2 = warper.bilinear_splatting(frame1, None, trans_depth1, flow12, None, grid, is_image=True)
        diff = frame2.astype('float') - warped_frame2.astype('float')
        poi2_mask = numpy.any(numpy.abs(diff) > self.intensity_threshold, axis=2) & mask2

        poi2_mask_mod = numpy.copy(poi2_mask)
        poi2_mask_mod[0, :] = 0
        poi2_mask_mod[-1, :] = 0
        poi2_mask_mod[:, 0] = 0
        poi2_mask_mod[:, -1] = 0
        tcx_floor = numpy.clip(numpy.floor(trans_coordinates[:, :, 0]), a_min=0, a_max=w - 1).astype('int')
        tcx_ceil = numpy.clip(numpy.ceil(trans_coordinates[:, :, 0]), a_min=0, a_max=w - 1).astype('int')
        tcy_floor = numpy.clip(numpy.floor(trans_coordinates[:, :, 1]), a_min=0, a_max=h - 1).astype('int')
        tcy_ceil = numpy.clip(numpy.ceil(trans_coordinates[:, :, 1]), a_min=0, a_max=h - 1).astype('int')
        poi1_mask = numpy.zeros_like(poi2_mask)
        poi1_mask = poi1_mask | poi2_mask_mod[tcy_floor, tcx_floor]
        poi1_mask = poi1_mask | poi2_mask_mod[tcy_floor, tcx_ceil]
        poi1_mask = poi1_mask | poi2_mask_mod[tcy_ceil, tcx_floor]
        poi1_mask = poi1_mask | poi2_mask_mod[tcy_ceil, tcx_ceil]

        a = (self.window_size - 1) // 2
        frame_windows, depth_windows, poi1_mask_windows = [], [], []
        h1 = h - self.window_size + 1
        w1 = w - self.window_size + 1
        for i in range(self.window_size):
            for j in range(self.window_size):
                frame_windows.append(frame1[i: i + h1, j: j + w1])
                depth_windows.append(depth1[i: i + h1, j: j + w1])
                poi1_mask_windows.append(poi1_mask[i: i + h1, j: j + w1])
        frame_central = frame1[a:-a, a:-a]
        frame_windows = numpy.stack(frame_windows)
        depth_windows = numpy.stack(depth_windows)
        poi1_mask_windows = numpy.stack(poi1_mask_windows)
        frame_diff = numpy.linalg.norm(frame_windows.astype('float') - frame_central.astype('float'), axis=3)
        depth_threshold = (numpy.min(depth_windows, axis=0) + numpy.max(depth_windows, axis=0)) / 2
        bg_mask = depth_windows >= depth_threshold[None]
        total_mask = poi1_mask_windows | bg_mask
        masked_frame_diff = numpy.ma.masked_array(frame_diff, total_mask)
        min_diff_locs = numpy.ma.argmin(masked_frame_diff, axis=0)
        matched_depth = depth_windows[min_diff_locs[None], numpy.arange(h1)[None, :, None], numpy.arange(w1)[None, None, :]][0]
        corrected_depth1 = numpy.copy(depth1)
        corrected_depth1[a:-a, a:-a] = matched_depth
        generated_data_dict['depth'] = poi1_mask * corrected_depth1 + (~poi1_mask) * depth1

        save_generated_data(generated_data_dict, new_paths_dict)
                
        return


def Veed_start_data_generation(configs: dict, mode: str, video_names: List[str], frame_nos: List[int]):
    
    generated_root_dirpath = Path(configs['database_dirpath']) / f'PreprocessedData/{mode}'
    input_root_dirpath = Path(configs['database_dirpath']) / f'RenderedData/{mode}'

    depth_corrector = DepthCorrector(intensity_threshold = 20, window_size = 11)
    warper = Warper(resolution=(1080, 1920), configs = configs, normalize_pos_vectors = False)
    intrinsic = warper.camera_intrinsic_transform()
    num_videos = len(video_names)

    for i, video_name in enumerate(video_names):

        input_video_dirpath = input_root_dirpath / video_name
        generated_video_dirpath = generated_root_dirpath / video_name

        for seq in range(4):

            input_video_seq_dirpath = input_video_dirpath / f'seq{seq:02}'
            generated_video_seq_dirpath = generated_video_dirpath / f'seq{seq:02}'

            new_subfolders_list = ['depth']
            
            created_folders_dict = create_newfolders(generated_video_seq_dirpath, None, None, new_subfolders_list)

            transformation_path = input_video_seq_dirpath / 'render/TransformationMatrices.csv'
            transformation_matrices = numpy.array(numpy.genfromtxt(transformation_path, delimiter=','))

            description = f'sequence{seq:02} {i + 1:03}/{num_videos:03} {video_name}'
            for frame1_num in tqdm(frame_nos, desc=description):
                frame2_num = frame1_num + 1

                new_paths_dict, continue_flag = create_newpaths(created_folders_dict, frame1_num, frame2_num)

                if continue_flag:
                    continue

                frame1_path = input_video_seq_dirpath / f'render/rgb/{frame1_num:04}.png'
                frame2_path = input_video_seq_dirpath / f'render/rgb/{frame2_num:04}.png'
                depth1_path = input_video_seq_dirpath / f'render/depth/{frame1_num:04}.exr'
                
                frame1 = read_image(frame1_path)
                frame2 = read_image(frame2_path)
                depth1 = read_depth(depth1_path)
                transformation1 = transformation_matrices[frame1_num - 1].reshape(4, 4)
                transformation2 = transformation_matrices[frame2_num - 1].reshape(4, 4)

                depth_corrector.correct_depth(warper, depth1, frame1, frame2, transformation1, transformation2, intrinsic, new_paths_dict)
    return


def SceneNet_start_data_generation(configs: dict, mode: str, frames_data: pandas.DataFrame, frame_indices: list):

    generated_root_dirpath = Path(configs['database_dirpath']) / f'PreprocessedData/{mode}'
    input_root_dirpath = Path(configs['database_dirpath']) / f'RenderedData/{mode}'

    depth_corrector = DepthCorrector(intensity_threshold = 20, window_size = 5)
    warper = Warper(resolution=(240, 320), configs = configs, normalize_pos_vectors = True)
    intrinsic = warper.camera_intrinsic_transform()
    num_frames = len(frames_data)

    prev_video_name = str(f'xxxxx')

    for row_num in tqdm(range(num_frames)):

        if row_num % 6 not in frame_indices:
            continue

        frame_data = frames_data.iloc[row_num]
        _, video_num, frame_num = frame_data

        video_num = int(video_num)
        video_name = str(f'{video_num:05}')
        frame1_num = int(frame_num)

        frame2_num = frame1_num + 25

        if prev_video_name != video_name:

            prev_video_name = (video_name + '.')[:-1]

            input_video_dirpath = input_root_dirpath / video_name
            generated_video_dirpath = generated_root_dirpath / video_name

            new_subfolders_list = ['depth']
            created_folders_dict = create_newfolders(generated_video_dirpath, None, None, new_subfolders_list)

            transformation_path = input_video_dirpath / 'render/TransformationMatrix.txt'
            transformation_matrices = numpy.array(numpy.genfromtxt(transformation_path, delimiter=','))
    
        new_paths_dict, continue_flag = create_newpaths(created_folders_dict, frame1_num, frame2_num)
        
        if continue_flag:
            continue

        frame1_path = input_video_dirpath / f'render/photo/{frame1_num:04}.jpg'
        frame2_path = input_video_dirpath / f'render/photo/{frame2_num:04}.jpg'
        depth1_path = input_video_dirpath / f'render/depth/{frame1_num:04}.png'
        
        frame1 = read_image(frame1_path)
        frame2 = read_image(frame2_path)
        depth1 = read_depth(depth1_path)
        transformation1 = transformation_matrices[frame1_num // 25].reshape(4, 4)
        transformation2 = transformation_matrices[frame2_num // 25].reshape(4, 4)

        depth_corrector.correct_depth(warper, depth1, frame1, frame2, transformation1, transformation2, intrinsic, new_paths_dict)
                
    return

# # the unexpected guest
