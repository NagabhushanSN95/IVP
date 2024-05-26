# Shree KRISHNAya Namaha

# import datetime
# import json
# import os
# import shutil
# import time
# import traceback
# from pathlib import Path
# from typing import Union

# import Imath
# import OpenEXR
# import numpy
# import pandas
# import simplejson
# import skimage.io
# import torch
# from matplotlib import pyplot
# from tqdm import tqdm

# from data_loaders import DataLoaderFactory
from Test_utils import *
import utils
# from decoders.DecoderFactory import get_decoder
# from infillers.InfillerFactory import get_infiller
# from read_off_estimators.ReadOffEstimatorFactory import get_read_off_estimation_network

this_filename = Path(__file__).stem


def Veed_start_testing(configs: dict, mode: str):
    
    test_num = configs['test_num']

    gt_videos_dirpath = Path(configs['database_dirpath']) / f'RenderedData/{mode}'
    output_dirpath = Path(f'../Runs/{mode}ing/{mode}{test_num:04}')
    output_dirpath.mkdir(parents=True, exist_ok=True)

    save_configs(output_dirpath, configs)

    proe_model = ProeModel(configs, device='gpu0')
    proe_model.load_model()
    proe_model.pre_test_ops()

    frame_nos = [2, 4, 6]
    videos_dirpaths = sorted(gt_videos_dirpath.iterdir(), key=lambda path: path.as_posix().lower())
    num_videos = len(videos_dirpaths)
    
    for i, video_dirpath in enumerate(videos_dirpaths):
        video_name = video_dirpath.stem

        for seq in range(4):
            video_seq_dirpath = video_dirpath / f'seq{seq:02}'
            pred_frames_dirpath = output_dirpath / video_name / f'seq{seq:02}/PredictedFrames'
            pred_frames_dirpath.mkdir(parents=True, exist_ok=True)
            
            for frame3_num in tqdm(frame_nos, desc=f'{i + 1:03}/{num_videos:03}. Video: {video_name} sequence{seq:02}'): 
                frame4_num = frame3_num + 1
                frame5_num = frame3_num + 2           
                
                frame3_path = video_seq_dirpath / f'render/rgb/{frame3_num:04}.png'
                frame5_path = video_seq_dirpath / f'render/rgb/{frame5_num:04}.png'

                output_frame3_path = pred_frames_dirpath / f'{frame3_num:04}.png'
                infilled_frame4_path = pred_frames_dirpath / f'{frame4_num:04}.png'
                output_frame5_path = pred_frames_dirpath / f'{frame5_num:04}.png'
                
                if not output_frame3_path.exists():
                    shutil.copyfile(frame3_path.as_posix(), output_frame3_path.as_posix())

                if not infilled_frame4_path.exists():
                    infilled_frame4 = proe_model.infer_frame(video_name, seq, frame3_num)
                    skimage.io.imsave(infilled_frame4_path, infilled_frame4, check_contrast=False)
                    
                if not output_frame5_path.exists():
                    shutil.copyfile(frame5_path.as_posix(), output_frame5_path.as_posix())
    return


def SceneNet_start_testing(configs: dict, mode: str):
    
    test_num = configs['test_num']

    gt_videos_dirpath = Path(configs['database_dirpath']) / f'RenderedData/{mode}'
    output_dirpath = Path(f'../Runs/{mode}ing/{mode}{test_num:04}')
    output_dirpath.mkdir(parents=True, exist_ok=True)

    save_configs(output_dirpath, configs)

    proe_model = ProeModel(configs, device='gpu0')
    proe_model.load_model()
    proe_model.pre_test_ops()

    frames_data_path = Path(configs['database_dirpath']) / f'{mode}Set.csv'
    frames_data = pandas.read_csv(frames_data_path.as_posix())
    num_frames = len(frames_data)
    
    for row_num in tqdm(range(12)):
        if row_num % 6 not in [2,3,4]:
            continue
        frame_data = frames_data.iloc[row_num]
        _, video_num, frame_num = frame_data

        video_num = int(video_num)
        video_name = str(f'{video_num:05}')
        frame3_num = int(frame_num)
        frame4_num = frame3_num + 1 * 25
        frame5_num = frame3_num + 2 * 25

        video_dirpath = gt_videos_dirpath / video_name
        pred_frames_dirpath = output_dirpath / video_name / f'PredictedFrames'
        pred_frames_dirpath.mkdir(parents=True, exist_ok=True)
        
        frame3_path = video_dirpath / f'render/photo/{frame3_num:04}.jpg'
        frame5_path = video_dirpath / f'render/photo/{frame5_num}.jpg'

        output_frame3_path = pred_frames_dirpath / f'{frame3_num:04}.png'
        infilled_frame4_path = pred_frames_dirpath / f'{frame4_num:04}.png'
        output_frame5_path = pred_frames_dirpath / f'{frame5_num:04}.png'
        
        if not output_frame3_path.exists():
            framee = utils.read_image(frame3_path)
            utils.save_image(output_frame3_path, framee)

        if not infilled_frame4_path.exists():
            infilled_frame4 = proe_model.infer_frame(video_name, None, frame3_num)
            skimage.io.imsave(infilled_frame4_path, infilled_frame4, check_contrast=False)
            
        if not output_frame5_path.exists():
            framee = utils.read_image(frame5_path)
            utils.save_image(output_frame5_path, framee)

    return



def main(args):
    
    with open(args.configs.as_posix(), 'r') as configs_file:
        configs = json.load(configs_file)

    if bool(configs):
        configs['test_num'] = args.test_num
        
        if args.generate_data:
            print("\nData generation starts....")
            run.run(configs, 'Test', args.correct_depth)
            print("\nData generation done....")
        
        if configs['database_name'] == 'Veed':
            Veed_start_testing(configs, 'Test')
        elif configs['database_name'] == 'SceneNet':
            SceneNet_start_testing(configs, 'Test')
            
    else:
        print("No training configs specified; exiting test code")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configs', type=Path, help='The path to the configs of the trained model to be tested.')
    parser.add_argument('-t', '--test_num', type=int, default=9999, help='The number for the test folder.')
    parser.add_argument('-data', '--generate_data', type=bool, default=False, help='If data has to be generated for the test set; default is False.')
    parser.add_argument('-depth_corr', '--correct_depth', type = bool, default=True)    # set this to false if you want to directly use rendered depth
    args = parser.parse_args()

    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main(args)
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))