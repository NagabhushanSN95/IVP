# Shree KRISHNAya Namaha

import datetime
import json
import os
import shutil
import time
import traceback
import argparse
from pathlib import Path
from typing import Union, Optional

import numpy
import pandas
import simplejson
import skimage.io
import torch
from tqdm import tqdm

from data_generators import run
from data_loaders.DataLoaderFactory import get_data_loader
from infillers.InfillerFactory import get_infiller
from read_off_estimators.ReadOffEstimatorFactory import get_read_off_estimation_network


class ProeModel:
    def __init__(self, configs: dict, mode: str = 'Test', device: Union[int, str] = 'gpu0'):
        self.infiller = None
        self.mode = mode
        self.configs = configs
        self.device = self.get_device(device)
        self.build_model()
        self.data_loader = get_data_loader(configs['database_name'], configs['database_dirpath'], self.mode)
        return

    @staticmethod
    def get_device(device: Union[int, str]):
        if device == 'cpu':
            device = torch.device('cpu')
        elif device.startswith('gpu') and torch.cuda.is_available():
            gpu_num = int(device[3:])
            device = torch.device(f'cuda:{gpu_num}')
        else:
            device = torch.device('cpu')
        return device

    def build_model(self):
        read_off_estimator = None
        decoder = None
        if 'read_off_estimator' in self.configs.keys():
            read_off_estimator = get_read_off_estimation_network(self.configs['read_off_estimator'])
        if 'decoder' in self.configs.keys():
            decoder = get_decoder(self.configs['decoder'])
        self.infiller = get_infiller(self.configs['infiller'], read_off_estimator).to(self.device)
        return

    def load_model(self, model_num: int = 0, cpu: bool = False):
        train_num = self.configs["train_num"]
        if not model_num:
            model_num = self.configs['num_epochs']
        full_model_name = f'Model_Epoch{model_num:03}.tar'
        train_dirpath = Path(f'../Runs/Training/Train{train_num:04}')
        saved_models_dirpath = train_dirpath / 'SavedModels'
        model_path = saved_models_dirpath / full_model_name
        if not cpu:
            checkpoint_state = torch.load(model_path, map_location=self.device)
        else:
            checkpoint_state = torch.load(model_path, map_location='cpu')
        epoch_num = checkpoint_state['epoch']
        self.infiller.load_state_dict(checkpoint_state['model_state_dict'])
        print(f'Loaded Model in Train{train_num:02}/{full_model_name} trained for {epoch_num} epochs')
        return

    def pre_test_ops(self):
        self.infiller.eval()
        return

    # def get_data_loader(self, database_dirpath, group):
    #     database_name = self.configs['database_name']
    #     data_loader = DataLoaderFactory.get_data_loader(database_name, database_dirpath, group, patch_size=None)
    #     return data_loader

    @staticmethod
    def move_to_device(tensors_dict: dict, device_):
        for key in tensors_dict.keys():
            tensors_dict[key] = tensors_dict[key].to(device_)
        return

    # noinspection PyUnusedLocal
    def infer_frame(self, video_name: str, seq: int, frame3_num: int):
        input_data = self.load_data(video_name, seq, frame3_num)
        self.move_to_device(input_data, self.device)
        with torch.no_grad():
            output_dict = self.infiller(input_data)

        processed_output = self.post_process_output(output_dict)
        infilled_frame4 = self.post_process_frame(processed_output['predicted_frame4'])
        return infilled_frame4

    def load_data(self, video_name, seq, frame3_num):
        if seq is None:
            input_data = self.data_loader.load_test_data(video_name, frame3_num)
        else:
            input_data = self.data_loader.load_test_data(video_name, seq, frame3_num)
        return input_data

    @staticmethod
    def post_process_output(output_batch: dict):
        processed_batch = {}
        for key in output_batch.keys():
            if isinstance(output_batch[key], torch.Tensor):
                processed_batch[key] = numpy.moveaxis(output_batch[key].detach().cpu().numpy(), [0, 1, 2, 3], [0, 3, 1, 2])[0]
        return processed_batch

    @staticmethod
    def post_process_frame(infilled_frame):
        processed_frame = numpy.round((infilled_frame + 1) * 255 / 2).astype('uint8')
        return processed_frame

    @staticmethod
    def post_process_mask(mask):
        processed_mask = mask[:,:,0].astype('uint8')
        return processed_mask

    @staticmethod
    def post_process_depth(depth):
        processed_depth = depth[:, :, 0]
        return processed_depth


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError('Configs mismatch while resuming testing')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return
