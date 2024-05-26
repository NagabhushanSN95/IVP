# Shree KRISHNAya Namaha
# Common utility functions for Data generation

import time
import numpy
import datetime
import traceback
from pathlib import Path
from typing import Optional, Tuple, List

import Imath
import OpenEXR
import skimage.io
from tqdm import tqdm


def read_image(path: Path) -> numpy.ndarray:
    image = skimage.io.imread(path.as_posix())[:, :, :3]
    return image


def save_image(path: Path, data: numpy.array):
    "data should be in range 0-255 and as uint8"
    skimage.io.imsave(path.as_posix(), data, check_contrast=False)
    return


def read_depth(path: Path) -> numpy.ndarray:
    if path.suffix == '.png':
        # SceneNet original depth
        depth = skimage.io.imread(path.as_posix()) * 0.001
        depth[depth == 0] = 1000
    elif path.suffix == '.npy':
        depth = numpy.load(path.as_posix())
    elif path.suffix == '.exr':
        exr_file = OpenEXR.InputFile(path.as_posix())
        raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
        depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
        height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
        width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
        depth = numpy.reshape(depth_vector, (height, width))
    else:
        raise RuntimeError(f'Unknown depth format: {path.suffix}')

    return depth.astype('float32')


def read_mask(path: Path):
    mask = skimage.io.imread(path.as_posix())

    unique_values = numpy.unique(mask)
    assert unique_values.size <= 2

    bool_mask = mask == 255
    return bool_mask


def read_npy(path: Path):
    return numpy.load(path).astype('float32')


def save_npy(path: Path, data: numpy.array):
    numpy.save(path.as_posix(), data)
    return


def create_newfolders(video_dirpath: Path, new_folder_name: str, num_steps: Optional[int], subfolder_list: List[str]):
    
    assert len(subfolder_list) != 0

    created_folders = dict()

    for subfolder_name in subfolder_list:

        if new_folder_name is None:     # Corrected depth
            new_dirpath = video_dirpath / subfolder_name
        elif num_steps is None:   # InfillingVector
            new_dirpath = video_dirpath / new_folder_name / subfolder_name
        else:   
            new_dirpath = video_dirpath / new_folder_name / f'{num_steps}step/{subfolder_name}'

        new_dirpath.mkdir(parents=True, exist_ok=True)
        created_folders[subfolder_name] = new_dirpath
    
    return created_folders


def create_newpaths(new_folders: dict, frame1_num: int, frame2_num: int):

    assert len(new_folders) != 0

    created_paths = dict()
    continue_flag = False

    for subfolder_name in new_folders.keys():

        if subfolder_name in ['warped_frames', 'masks', 'disocclusion_masks']:
            new_path = new_folders[subfolder_name] / f'{frame2_num:04}.png'
            created_paths[subfolder_name] = new_path

        elif subfolder_name in ['transformed_points', 'estimated_iv', 'depth']:
            new_path = new_folders[subfolder_name] / f'{frame1_num:04}.npy'
            created_paths[subfolder_name] = new_path

        elif subfolder_name in ['warped_depths', 'warped_iv']:
            new_path = new_folders[subfolder_name] / f'{frame2_num:04}.npy'
            created_paths[subfolder_name] = new_path

    for pathh in created_paths.values():
        if pathh.exists():
            continue_flag = True
            break
    
    return created_paths, continue_flag


def save_generated_data(generated_data_dict: dict, save_paths: dict):
    for data_name in save_paths.keys():
        if data_name in ['warped_frames', 'masks', 'disocclusion_masks']:
            save_image(save_paths[data_name], generated_data_dict[data_name])
        elif data_name in ['warped_depths', 'transformed_points', 'estimated_iv', 'warped_iv', 'depth']:
            save_npy(save_paths[data_name], generated_data_dict[data_name])
            
    return

# the unexpected guest