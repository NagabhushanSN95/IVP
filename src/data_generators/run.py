import numpy
import pandas
import traceback
from pathlib import Path
from typing import Optional, Tuple, List

import data_generators.PoseWarping
import data_generators.IVPrior
import data_generators.DepthCorrector


def run(configs: dict, mode: str,  correct_depth: bool):
    """
    mode: train or test, depending on the file that is calling this function
    corrected_depth: bool- set this to false if you want to directly use rendered depth
    """

    input_root_dirpath = Path(configs['database_dirpath']) / f'RenderedData/{mode}'
    
    num_steps1 = 2
    num_steps2 = 1
    # IV is estimated for frame n that is 2-step-warped from n-2. Hence, num_steps1 = 2. 
    # IV estimated in previous step is 1-step-warped to view of frame n+1. hence, num_steps2 = 1. 

    if configs['database_name'] == 'Veed':

        input_video_names = [pathh.stem for pathh in sorted(input_root_dirpath.iterdir())][:2]

        if mode == 'Test':
            frame_nos = [2, 4, 6]
        elif mode == 'Train':
            frame_nos = list(range(2, 11))
        
        
        if correct_depth:
            frame_nos_2 = list(range(frame_nos[0])) + frame_nos
            data_generators.DepthCorrector.Veed_start_data_generation(configs, mode, input_video_names, frame_nos_2)
        data_generators.PoseWarping.Veed_start_data_generation(configs, mode, input_video_names, frame_nos, num_steps1)
        data_generators.PoseWarping.Veed_start_data_generation(configs, mode, input_video_names, frame_nos, num_steps2)
        data_generators.IVPrior.Veed_start_data_generation(configs, mode, input_video_names, frame_nos, num_steps1, num_steps2)

    elif configs['database_name'] == 'SceneNet':

        frame_indices = [2,3,4]

        if mode == 'Test':
            frames_data_path = Path(configs['database_dirpath']) / f'TestSet.csv'
        elif mode == 'Train':
            frames_data_path = Path(configs['database_dirpath']) / f'TrainSet.csv'

        frames_data = pandas.read_csv(frames_data_path.as_posix())[:12]
        
        if correct_depth:
            frame_indices_2 = list(range(frame_indices[0])) + frame_indices
            data_generators.DepthCorrector.SceneNet_start_data_generation(configs, mode, frames_data, frame_indices_2)
        data_generators.PoseWarping.SceneNet_start_data_generation(configs, mode, frames_data, frame_indices, num_steps1)
        data_generators.PoseWarping.SceneNet_start_data_generation(configs, mode, frames_data, frame_indices, num_steps2)
        data_generators.IVPrior.SceneNet_start_data_generation(configs, mode, frames_data, frame_indices, num_steps1, num_steps2)