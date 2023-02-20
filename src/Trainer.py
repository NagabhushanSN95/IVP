# Shree KRISHNAya Namaha

import os
import json
import time
import random
import datetime
import traceback
import argparse
from pathlib import Path

import torch
import numpy
import pandas
import simplejson
import skimage.io
from tqdm import tqdm
from matplotlib import pyplot
from torch.utils.data.dataloader import DataLoader

from data_generators import run
from infillers.InfillerFactory import get_infiller
from loss_functions.LossComputer02 import LossComputer
from data_loaders.DataLoaderFactory import get_data_loader
from read_off_estimators.ReadOffEstimatorFactory import get_read_off_estimation_network


this_filename = Path(__file__).stem
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def train_one_epoch(infiller, train_data_loader, loss_computer: LossComputer, optimizer):
    def update_losses_dict(epoch_losses_dict_: dict, iter_losses_dict_: dict):
        if epoch_losses_dict_ is None:
            epoch_losses_dict_ = {}
            for loss_name in iter_losses_dict_.keys():
                epoch_losses_dict_[loss_name] = iter_losses_dict_[loss_name].item()
        else:
            for loss_name in epoch_losses_dict_.keys():
                epoch_losses_dict_[loss_name] += iter_losses_dict_[loss_name].item()
        return epoch_losses_dict_

    epoch_losses_dict = None

    for iter_num, input_batch in enumerate(tqdm(train_data_loader, leave=False)):
        move_to_device(input_batch, device)
        optimizer.zero_grad()
        output_batch = infiller(input_batch)
        iter_losses_dict = loss_computer.compute_losses(input_batch, output_batch)
        iter_losses_dict['Total Loss'].backward()
        optimizer.step()
        epoch_losses_dict = update_losses_dict(epoch_losses_dict, iter_losses_dict)
    return epoch_losses_dict


def validation_one_epoch(infiller, val_data_loader, loss_computer):
    def update_losses_dict_(epoch_losses_dict_: dict, iter_losses_dict_: dict):
        if epoch_losses_dict_ is None:
            epoch_losses_dict_ = {}
            for loss_name in iter_losses_dict_.keys():
                epoch_losses_dict_[loss_name] = iter_losses_dict_[loss_name].item()
        else:
            for loss_name in epoch_losses_dict_.keys():
                epoch_losses_dict_[loss_name] += iter_losses_dict_[loss_name].item()
        return epoch_losses_dict_

    epoch_losses_dict = None
    infiller.eval()
    with torch.no_grad():
        for iter_num, input_batch in enumerate(tqdm(val_data_loader, leave=False)):
            move_to_device(input_batch, device)
            output_batch = infiller(input_batch)
            iter_losses_dict = loss_computer.compute_losses(input_batch, output_batch)
            epoch_losses_dict = update_losses_dict_(epoch_losses_dict, iter_losses_dict)
            if (iter_num + 1) * val_data_loader.batch_size >= 500:
                break
    infiller.train()
    return epoch_losses_dict


def test_one_batch(infiller, input_batch, device1, return_infill_mask: bool = False, return_infilling_vectors: bool = False):
    move_to_device(input_batch, device1)
    with torch.no_grad():
        output_batch = infiller(input_batch)
    infilled_batch = output_batch['predicted_frame4']
    return_args = [infilled_batch]
    if return_infill_mask:
        return_args.append(output_batch['infill_mask4'])
    if return_infilling_vectors:
        return_args.append(output_batch['read_off_values'])
    if len(return_args) == 1:
        return_args = return_args[0]
    return return_args


def train(infiller, train_dataset, val_dataset, loss_computer, optimizer, num_epochs: int, batch_size: int,
          output_dirpath: Path,
          sample_save_interval: int, model_save_interval: int):
    def update_losses_data(epoch_num_: int, epoch_losses_: dict, cumulative_losses_: pandas.DataFrame, save_path: Path):
        curr_time = datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p')
        epoch_data = [epoch_num_, curr_time] + list(epoch_losses_.values())
        if cumulative_losses_ is None:
            loss_names = list(epoch_losses_.keys())
            column_names = ['Epoch Num', 'Time'] + loss_names
            cumulative_losses_ = pandas.DataFrame([epoch_data], columns=column_names)
        else:
            num_curr_rows = cumulative_losses_.shape[0]
            cumulative_losses_.loc[num_curr_rows] = epoch_data
        cumulative_losses_.to_csv(save_path, index=False)
        return cumulative_losses_

    def print_losses(epoch_num_: int, epoch_losses_: dict, train_: bool):
        curr_time = datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p')
        if train_:
            log_string = 'Train '
        else:
            log_string = 'Validation '
        log_string += f'{epoch_num_:03}: {curr_time}; '
        for loss_name in epoch_losses_.keys():
            log_string += f'{loss_name}: {epoch_losses_[loss_name]:0.04f}; '
        print(log_string, flush=True)
        time.sleep(1)
        return

    print('Training begins... (not Batman Begins)')
    loss_plots_dirpath = output_dirpath / 'LossPlots'
    sample_images_dirpath = output_dirpath / 'Samples'
    saved_models_dirpath = output_dirpath / 'SavedModels'
    loss_plots_dirpath.mkdir(exist_ok=True)
    sample_images_dirpath.mkdir(exist_ok=True)
    saved_models_dirpath.mkdir(exist_ok=True)

    train_losses_path = loss_plots_dirpath / 'TrainLosses.csv'
    val_losses_path = loss_plots_dirpath / 'ValidationLosses.csv'
    train_losses_data = pandas.read_csv(train_losses_path) if train_losses_path.exists() else None
    val_losses_data = pandas.read_csv(val_losses_path) if val_losses_path.exists() else None

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                   num_workers=4)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                 num_workers=4)
    start_epoch_num = load_model(infiller, optimizer, saved_models_dirpath)
    for epoch_num in range(start_epoch_num, num_epochs):
        epoch_train_loss = train_one_epoch(infiller=infiller, train_data_loader=train_data_loader,
                                           loss_computer=loss_computer, optimizer=optimizer)
        train_losses_data = update_losses_data(epoch_num + 1, epoch_train_loss, train_losses_data, train_losses_path)
        print_losses(epoch_num + 1, epoch_train_loss, train_=True)
        epoch_val_loss = validation_one_epoch(infiller=infiller, val_data_loader=val_data_loader,
                                              loss_computer=loss_computer)
        val_losses_data = update_losses_data(epoch_num + 1, epoch_val_loss, val_losses_data, val_losses_path)
        print_losses(epoch_num + 1, epoch_val_loss, train_=False)

        if (epoch_num + 1) % sample_save_interval == 0:
            save_sample_images(epoch_num + 1, sample_images_dirpath, infiller, val_data_loader)

        if (epoch_num + 1) % model_save_interval == 0:
            save_model(infiller, optimizer, epoch_num + 1, saved_models_dirpath)

        # Save model after every epoch
        save_model(infiller, optimizer, epoch_num + 1, saved_models_dirpath, label='Latest')
    save_plots(loss_plots_dirpath, train_losses_path, prefix='Train')
    save_plots(loss_plots_dirpath, val_losses_path, prefix='Validation')
    return


def move_to_device(tensors_dict: dict, device_):
    for key in tensors_dict.keys():
        tensors_dict[key] = tensors_dict[key].to(device_)
    return


def save_sample_images(epoch_num, save_dirpath, infiller, val_data_loader):
    def convert_tensor_to_image(tensor_batch_):
        np_array = tensor_batch_.detach().cpu().numpy()
        image_batch = ((numpy.moveaxis(np_array, [0, 1, 2, 3], [0, 3, 1, 2]) + 1) * 255 / 2).astype('uint8')
        return image_batch

    infiller.eval()
    with torch.no_grad():
        input_batch = next(val_data_loader.__iter__())
        move_to_device(input_batch, device)
        warped_batch = input_batch['warped_frame4']
        true_batch = input_batch['frame4']
        output_batch = infiller(input_batch)
        infilled_batch = output_batch['predicted_frame4']
        sample_images = [warped_batch, infilled_batch, true_batch]
    infiller.train()

    for i in range(3):
        # noinspection PyTypeChecker
        numpy_batch = convert_tensor_to_image(sample_images[i])
        sample_images[i] = numpy.concatenate(numpy_batch, axis=1)
    sample_collage = numpy.concatenate(sample_images, axis=0)
    save_path = save_dirpath / f'Epoch_{epoch_num:03}.png'
    skimage.io.imsave(save_path.as_posix(), sample_collage)
    return


def save_model(model, optimizer, epoch_num: int, save_dirpath: Path, label: str = None):
    if label is None:
        label = f'Epoch{epoch_num:03}'
    save_path = save_dirpath / f'Model_{label}.tar'
    checkpoint_state = {
        'epoch': epoch_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint_state, save_path)
    return


def load_model(model, optimizer, saved_models_dirpath: Path):
    latest_model_path = saved_models_dirpath / 'Model_Latest.tar'
    if latest_model_path.exists():
        checkpoint_state = torch.load(latest_model_path)
        epoch_num = checkpoint_state['epoch']
        model.load_state_dict(checkpoint_state['model_state_dict'])
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        print(f'Resuming Training from epoch {epoch_num + 1}')
    else:
        epoch_num = 0
    return epoch_num


def save_plots(save_dirpath: Path, loss_data_path: Path, prefix):
    loss_data = pandas.read_csv(loss_data_path)
    epoch_nums = loss_data['Epoch Num']
    for loss_name in loss_data.keys()[2:]:
        loss_values = loss_data[loss_name]
        save_path = save_dirpath / f'{prefix}_{loss_name}.png'
        pyplot.plot(epoch_nums, loss_values)
        pyplot.savefig(save_path)
        pyplot.close()
    return


def init_seeds(seed: int = 1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    return


def save_configs(output_dirpath: Path, configs: dict):
    # Save configs
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        # If resume_training is false, an error would've been raised when creating output directory. No need to handle
        # it here.
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        configs['seed'] = old_configs['seed']
        if configs['num_epochs'] > old_configs['num_epochs']:
            # configs['num_epochs'] = old_configs['num_epochs']
            old_configs['num_epochs'] = configs['num_epochs']
        if configs != old_configs:
            raise RuntimeError('Configs mismatch while resuming training')
    else:
        with open(configs_path.as_posix(), 'w') as configs_file:
            simplejson.dump(configs, configs_file, indent=4)
    return


def print_configs(configs:dict):
    for keyy in configs.keys():
        print(keyy, ": ", configs[keyy])
        
    return


def start_training(configs: dict):

    # Setup output dirpath
    output_dirpath = Path(f'../Runs/Training/Train{configs["train_num"]:04}')
    output_dirpath.mkdir(parents=True, exist_ok=configs['resume_training'])
    save_configs(output_dirpath, configs)
    init_seeds(configs['seed'])

    # Create data_loaders, models, optimizers etc
    train_dataset = get_data_loader(configs['database_name'], configs['database_dirpath'], split_name='Train',
                                    patch_size=configs['patch_size'])
    val_dataset = get_data_loader(configs['database_name'], configs['database_dirpath'], split_name='Validation',
                                  patch_size=configs['patch_size'])
    read_off_estimator = get_read_off_estimation_network(configs['read_off_estimator'])
    infiller = get_infiller(configs['infiller'], read_off_estimator).to(device)
    loss_computer = LossComputer(configs['losses'])
    optimizer = torch.optim.Adam(list(infiller.parameters()), lr=configs['lr'],
                                 betas=(configs['beta1'], configs['beta2']))

    # Save the names of all train videos used.
    train_video_names = train_dataset.get_video_names()
    train_video_names = [name + '\n' for name in train_video_names]
    train_video_names_path = output_dirpath / 'TrainVideoNames.txt'
    with open(train_video_names_path.as_posix(), 'w') as output_file:
        output_file.writelines(train_video_names)
    val_video_names = val_dataset.get_video_names()
    val_video_names = [name + '\n' for name in val_video_names]
    val_video_names_path = output_dirpath / 'ValidationVideoNames.txt'
    with open(val_video_names_path.as_posix(), 'w') as output_file:
        output_file.writelines(val_video_names)

    # Start training
    train(infiller=infiller, train_dataset=train_dataset, val_dataset=val_dataset, loss_computer=loss_computer,
          optimizer=optimizer, num_epochs=configs['num_epochs'], batch_size=configs['batch_size'],
          output_dirpath=output_dirpath, sample_save_interval=configs['sample_save_interval'],
          model_save_interval=configs['model_save_interval'])
    return


def main(args):

    args.configs = Path(args.configs)
    with open(args.configs.as_posix(), 'r') as configs_file:
        configs = json.load(configs_file)

    if bool(configs):
        print_configs(configs)
        
        if args.generate_data:
            print("\nData generation starts....")
            run.run(configs, 'Train', args.correct_depth)
            print("\nData generation done....")
        
        start_training(configs)
    else:
        print("No training configs specified; exiting training code")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configs', type = str, default = 'configs/Configs_VEED.json')
    parser.add_argument('-data', '--generate_data', type = bool, default=False)
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