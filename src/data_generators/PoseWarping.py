# Shree KRISHNAya Namaha
# Based on https://github.com/NagabhushanSN95/Pose-Warping/blob/main/src/Warper.py v-19/06/2021

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

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class Warper:
    def __init__(self, resolution: tuple, configs: dict, normalize_pos_vectors: bool = False):
        self.resolution = resolution
        self.h, self.w = self.resolution
        self.configs = configs
        self.normalize_pos_vectors = normalize_pos_vectors
        return


    def compute_transformed_points(self, depth1: numpy.ndarray, mask1: Optional[numpy.ndarray],
                                   transformation1: numpy.ndarray,
                                   transformation2: numpy.ndarray, intrinsic1: numpy.ndarray,
                                   intrinsic2: Optional[numpy.ndarray]):
        """
        Computes transformed position for each pixel location
        """
        h, w = depth1.shape
        depth1 = numpy.clip(depth1, a_min=0, a_max=1e10)
        if intrinsic2 is None:
            intrinsic2 = numpy.copy(intrinsic1)
        transformation = numpy.matmul(transformation2, numpy.linalg.inv(transformation1))

        y1d = numpy.array(range(h))
        x1d = numpy.array(range(w))
        x2d, y2d = numpy.meshgrid(x1d, y1d)
        ones_2d = numpy.ones(shape=(h, w))
        ones_4d = ones_2d[:, :, None, None]
        pos_vectors_homo = numpy.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]

        intrinsic1_inv = numpy.linalg.inv(intrinsic1)
        intrinsic1_inv_4d = intrinsic1_inv[None, None]
        intrinsic2_4d = intrinsic2[None, None]
        depth_4d = depth1[:, :, None, None]
        trans_4d = transformation[None, None]

        unnormalized_pos = numpy.matmul(intrinsic1_inv_4d, pos_vectors_homo)
        if not self.normalize_pos_vectors:
            world_points = depth_4d * unnormalized_pos
        else:
            unit_pos_vecs = unnormalized_pos / numpy.linalg.norm(unnormalized_pos, axis=2, keepdims=True)
            world_points = depth_4d * unit_pos_vecs
            
        world_points_homo = numpy.concatenate([world_points, ones_4d], axis=2)
        trans_world_homo = numpy.matmul(trans_4d, world_points_homo)
        trans_world = trans_world_homo[:, :, :3]
        trans_norm_points = numpy.matmul(intrinsic2_4d, trans_world)
        return trans_norm_points

    @staticmethod
    def bilinear_splatting(frame1: numpy.ndarray, mask1: Optional[numpy.ndarray], depth1: numpy.ndarray,
                           flow12: numpy.ndarray, flow12_mask: Optional[numpy.ndarray], grid: numpy.ndarray, is_image: bool = False) -> \
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Using inverse bilinear interpolation based splatting
        :param frame1: (h, w, c)
        :param mask1: (h, w): True if known and False if unknown. Optional
        :param depth1: (h, w)
        :param flow12: (h, w, 2)
        :param flow12_mask: (h, w): True if valid and False if invalid. Optional
        :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
        :return: warped_frame2: (h, w, c)
                 mask2: (h, w): True if known and False if unknown
        """
        h, w, c = frame1.shape
        if mask1 is None:
            mask1 = numpy.ones(shape=(h, w), dtype=bool)
        if flow12_mask is None:
            flow12_mask = numpy.ones(shape=(h, w), dtype=bool)
        # grid = self.create_grid(h, w)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = numpy.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = numpy.ceil(trans_pos_offset).astype('int')
        trans_pos_floor[:, :, 0] = numpy.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, 1] = numpy.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, 0] = numpy.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, 1] = numpy.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

        prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

        sat_depth1 = numpy.clip(depth1, a_min=0, a_max=1000)
        log_depth1 = numpy.log(1 + sat_depth1)
        depth_weights = numpy.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = prox_weight_nw * mask1 * flow12_mask / depth_weights
        weight_sw = prox_weight_sw * mask1 * flow12_mask / depth_weights
        weight_ne = prox_weight_ne * mask1 * flow12_mask / depth_weights
        weight_se = prox_weight_se * mask1 * flow12_mask / depth_weights

        weight_nw_3d = weight_nw[:, :, None]
        weight_sw_3d = weight_sw[:, :, None]
        weight_ne_3d = weight_ne[:, :, None]
        weight_se_3d = weight_se[:, :, None]

        warped_image = numpy.zeros(shape=(h + 2, w + 2, c), dtype=numpy.float32)
        warped_weights = numpy.zeros(shape=(h + 2, w + 2), dtype=numpy.float32)

        numpy.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
        numpy.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
        numpy.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
        numpy.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)

        numpy.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
        numpy.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
        numpy.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
        numpy.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)

        cropped_warped_image = warped_image[1:-1, 1:-1]
        cropped_weights = warped_weights[1:-1, 1:-1]

        mask = cropped_weights > 0
        with numpy.errstate(invalid='ignore'):
            warped_frame2 = numpy.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)

        if is_image:
            assert numpy.min(warped_frame2) >= 0
            assert numpy.max(warped_frame2) <= 256
            clipped_image = numpy.clip(warped_frame2, a_min=0, a_max=255)
            warped_frame2 = numpy.round(clipped_image).astype('uint8')
        return warped_frame2, mask

    
    @staticmethod    
    def create_grid(h, w):
        x_1d = numpy.arange(0, w)[None]
        y_1d = numpy.arange(0, h)[:, None]
        x_2d = numpy.repeat(x_1d, repeats=h, axis=0)
        y_2d = numpy.repeat(y_1d, repeats=w, axis=1)
        grid = numpy.stack([x_2d, y_2d], axis=2)
        return grid

    @staticmethod
    def convert_trans_pos_to_flow(trans_pos: numpy.ndarray):
        h, w = trans_pos.shape[:2]
        x_1d = numpy.arange(0, w)[None]
        y_1d = numpy.arange(0, h)[:, None]
        x_2d = numpy.repeat(x_1d, repeats=h, axis=0)
        y_2d = numpy.repeat(y_1d, repeats=w, axis=1)
        flow_x = trans_pos[:, :, 0] - x_2d
        flow_y = trans_pos[:, :, 1] - y_2d
        flow = numpy.stack([flow_x, flow_y], axis=2)
        return flow

    @staticmethod
    def compute_disocclusion_mask(mask: numpy.ndarray):
        col_cum_sum = numpy.cumsum(mask, axis=0)
        row_cum_sum = numpy.cumsum(mask, axis=1)

        last_col = row_cum_sum[:, -1]
        last_row = col_cum_sum[-1, :]

        new_region_mask = (col_cum_sum == 0) | (col_cum_sum == last_row[None]) | \
                          (row_cum_sum == 0) | (row_cum_sum == last_col[:, None])
        new_region_mask[-1, :] = col_cum_sum[-2, :] == col_cum_sum[-1, :]
        new_region_mask[:, -1] = row_cum_sum[:, -2] == row_cum_sum[:, -1]
        new_region_mask[-1, -1] = (col_cum_sum[-2, -1] == col_cum_sum[-1, -1]) | \
                                  (row_cum_sum[-1, -2] == row_cum_sum[-1, -1])
        disocclusion_mask = ~new_region_mask & ~mask
        return disocclusion_mask

    # @staticmethod
    def camera_intrinsic_transform(self, patch_start_point: tuple = (0, 0)):
        start_y, start_x = patch_start_point
        camera_intrinsics = numpy.eye(3)

        if self.configs['database_name'] == 'Veed':
            camera_intrinsics[0, 0] = 2100
            camera_intrinsics[0, 2] = self.resolution[1] / 2.0 - start_x
            camera_intrinsics[1, 1] = 2100
            camera_intrinsics[1, 2] = self.resolution[0] / 2.0 - start_y

        elif self.configs['database_name'] == 'SceneNet':
            vfov = 45
            hfov = 60
            camera_intrinsics[0, 0] = (self.resolution[1] / 2.0) / math.tan(math.radians(hfov / 2.0))
            camera_intrinsics[0, 2] = self.resolution[1] / 2.0
            camera_intrinsics[1, 1] = (self.resolution[0] / 2.0) / math.tan(math.radians(vfov / 2.0))
            camera_intrinsics[1, 2] = self.resolution[0] / 2.0

        return camera_intrinsics


    def forward_warp(self, new_subfolders_list: List, new_paths_dict: dict, frame1: numpy.ndarray, depth1: numpy.ndarray,
                        transformation1: numpy.ndarray, transformation2: numpy.ndarray, intrinsic1: numpy.ndarray, intrinsic2: numpy.ndarray=None):

        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        :param frame1: (h, w, 3) uint8 numpy array
        :param mask1: (h, w) bool numpy array. Wherever mask1 is False, those pixels are ignored while warping. Optional
        :param depth1: (h, w) float numpy array.
        :param transformation1: (4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (3, 3) camera intrinsic matrix
        :param intrinsic2: (3, 3) camera intrinsic matrix. Optional
        """

        generated_data_dict = dict()

        trans_points12 = self.compute_transformed_points(depth1, None, transformation1, transformation2, intrinsic1, intrinsic2)
        generated_data_dict['transformed_points'] = trans_points12.copy()
        trans_coordinates12 = trans_points12[:, :, :2, 0] / trans_points12[:, :, 2:3, 0]
        trans_depth1 = trans_points12[:, :, 2, 0]
        
        h, w = depth1.shape
        grid = self.create_grid(h, w)
        flow12 = trans_coordinates12 - grid
        
        generated_data_dict['warped_frames'], mask2 = self.bilinear_splatting(frame1, None, trans_depth1, flow12, None, grid, is_image=True)
        generated_data_dict['masks'] = (mask2.astype('uint8') * 255).copy()
        
        if 'disocclusion_masks' in new_subfolders_list:
            generated_data_dict['disocclusion_masks'] = self.compute_disocclusion_mask(mask2).astype('uint8') * 255
            
        if 'warped_depths' in new_subfolders_list:
            generated_data_dict['warped_depths'] = self.bilinear_splatting(trans_depth1[:, :, None], None, trans_depth1, flow12, None,
                                                  grid, is_image=False)[0][:, :, 0].copy()

        save_generated_data(generated_data_dict, new_paths_dict)

        return


def Veed_start_data_generation(configs: dict, mode: str, video_names: List[str], frame_nos: List[int], num_steps: int):
    
    generated_root_dirpath = Path(configs['database_dirpath']) / f'PrecomputedPriors/{mode}'
    input_root_dirpath = Path(configs['database_dirpath']) / f'RenderedData/{mode}'

    warper = Warper(resolution=(1080, 1920), configs = configs, normalize_pos_vectors = False)
    intrinsic = warper.camera_intrinsic_transform()
    num_videos = len(video_names)

    for i, video_name in enumerate(video_names):

        input_video_dirpath = input_root_dirpath / video_name
        generated_video_dirpath = generated_root_dirpath / video_name

        for seq in range(4):

            input_video_seq_dirpath = input_video_dirpath / f'seq{seq:02}'
            generated_video_seq_dirpath = generated_video_dirpath / f'seq{seq:02}'

            if num_steps == 1:
                new_subfolders_list = ['warped_frames', 'warped_depths', 'masks', 'transformed_points']
            else:
                new_subfolders_list = ['warped_frames', 'masks']

            created_folders_dict = create_newfolders(generated_video_seq_dirpath, 'PoseWarping', num_steps, new_subfolders_list)

            transformation_path = input_video_seq_dirpath / 'render/TransformationMatrices.csv'
            transformation_matrices = numpy.array(numpy.genfromtxt(transformation_path, delimiter=','))

            for frame1_num in tqdm(frame_nos, desc=f'sequence{seq:02} {i + 1:03}/{num_videos:03}: {video_name}'):
                
                if num_steps == 1:
                    frame2_num = frame1_num + num_steps

                elif num_steps == 2:
                    frame2_num = frame1_num
                    frame1_num -= num_steps

                frame1_path = input_video_seq_dirpath / f'render/rgb/{frame1_num:04}.png'
                depth1_path = Path(configs['database_dirpath']) / f'PreprocessedData/{mode}/{video_name}/seq{seq:02}/depth/{frame1_num:04}.npy'
                # If using rendered depth, use the foll path
                # depth1_path = input_video_seq_dirpath / f'render/depth/{frame1_num:04}.exr'

                new_paths_dict, continue_flag = create_newpaths(created_folders_dict, frame1_num, frame2_num)

                if continue_flag:
                    continue

                frame1 = read_image(frame1_path)
                depth1 = read_depth(depth1_path)
                transformation1 = transformation_matrices[frame1_num - 1].reshape(4, 4)
                transformation2 = transformation_matrices[frame2_num - 1].reshape(4, 4)

                warper.forward_warp(new_subfolders_list, new_paths_dict, frame1, depth1, transformation1, transformation2, intrinsic)
                
    return


def SceneNet_start_data_generation(configs: dict, mode: str, frames_data: pandas.DataFrame, frame_indices: list, num_steps: int):

    generated_root_dirpath = Path(configs['database_dirpath']) / f'PrecomputedPriors/{mode}'
    input_root_dirpath = Path(configs['database_dirpath']) / f'RenderedData/{mode}'

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

        if num_steps == 1:
            frame2_num = frame1_num + (num_steps*25)

        elif num_steps == 2:
            frame2_num = frame1_num
            frame1_num -= (num_steps*25)


        if prev_video_name != video_name:

            prev_video_name = (video_name + '.')[:-1]

            input_video_dirpath = input_root_dirpath / video_name
            generated_video_dirpath = generated_root_dirpath / video_name

            if num_steps == 1:
                new_subfolders_list = ['warped_frames', 'warped_depths', 'masks', 'transformed_points', 'disocclusion_masks']
            else:
                new_subfolders_list = ['warped_frames', 'masks', 'disocclusion_masks']
            created_folders_dict = create_newfolders(generated_video_dirpath, 'PoseWarping', num_steps, new_subfolders_list)

            transformation_path = input_video_dirpath / 'render/TransformationMatrix.txt'
            transformation_matrices = numpy.array(numpy.genfromtxt(transformation_path, delimiter=','))
    
        frame1_path = input_video_dirpath / f'render/photo/{frame1_num:04}.jpg'
        depth1_path = Path(configs['database_dirpath']) / f'PreprocessedData/{mode}/{video_name}/depth/{frame1_num:04}.npy'
        # If using corrected depth, use the foll path
        # depth1_path = input_video_dirpath / f'render/depth/{frame1_num:04}.png'

        new_paths_dict, continue_flag = create_newpaths(created_folders_dict, frame1_num, frame2_num)

        if continue_flag:
            continue

        print("\n", video_name, frame1_num)
        frame1 = read_image(frame1_path)
        depth1 = read_depth(depth1_path)
        transformation1 = transformation_matrices[frame1_num // 25].reshape(4, 4)
        transformation2 = transformation_matrices[frame2_num // 25].reshape(4, 4)

        warper.forward_warp(new_subfolders_list, new_paths_dict, frame1, depth1, transformation1, transformation2, intrinsic)
                
    return

# # the unexpected guest