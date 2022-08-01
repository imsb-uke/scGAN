#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import ntpath
import os
from multiprocessing import Pool, Manager
from shutil import copyfile

from estimators.run_exp import run_exp
from preprocessing.write_tfrecords import process_files

if __name__ == '__main__':
    """
    Main script to process the data and or start the training or 
    generate cells from an existing model
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--param', required=True,
                        help='Path to the parameters json file')

    parser.add_argument(
        '--process', required=False,
        default=False, action='store_true',
        help='process the raw file and generate TF records for training')

    parser.add_argument(
        '--train', required=False,
        default=False, action='store_true',
        help='Use the generated TF records for training the model')

    parser.add_argument(
        '--generate', required=False,
        default=False, action='store_true',
        help='generate will be used to generate in-silico cells')

    parser.add_argument(
        '--cells_no', required=False, nargs='+',
        help=' The cells number to be generated if the scGAN model is trained.'
             ' Cells number per cluster if the cscGAN model is trained.'
             ' If the scRNA data has 10 clusters, 10 integers should be passed.')

    parser.add_argument(
        '--model_path', required=False,
        help=' The path to the trained model folder. '
             'This folder must include the trained model '
             'job folder as well as the parameters json file.')

    parser.add_argument('--save_path', required=False,
                        help=' The path where the generated cells will be saved')

    a = parser.parse_args()

    # read experiments parameters file
    with open(a.param, 'r') as fp:
        parameters = json.load(fp)

    all_exp_dir = parameters['exp_param']['experiments_dir']
    GPU_NB = parameters['exp_param']['GPU']
    experiments = parameters['experiments']

    # loop over the different experiments specified in the json file
    exp_folders = []
    for exp in experiments:
        exp_param = experiments[exp]

        exp_dir = os.path.join(all_exp_dir, exp)
        raw_input = exp_param['input_ds']['raw_input']
        raw_file_name = ntpath.basename(raw_input)

        if a.process:
            try:
                os.makedirs(exp_dir)
            except OSError:
                raise OSError('The selected experiment folder already exists, '
                              'please remove it or select new one.')

            # copy raw file in every experiment folder
            copyfile(raw_input, os.path.join(exp_dir, raw_file_name))

            # create param.json file in every experiment folder
            with open(os.path.join(exp_dir, 'parameters.json'), 'w') as fp:
                fp.write(json.dumps(exp_param, sort_keys=True, indent=4))

        exp_folders.append(exp_dir)

    if a.process:
        process_files(exp_folders)

    if a.train:
        # create a queue with jobs and train models in parallel on
        # separate GPUs using multiprocessing package
        manager = Manager()
        avail_gpus = manager.list(GPU_NB)
        po = Pool(len(GPU_NB))
        r = po.map_async(run_exp,
                         ((exp_folder, avail_gpus) for exp_folder in exp_folders))
        r.wait()
        po.close()
        po.join()

    if a.generate:
        run_exp((a.model_path, [0]), mode='generate', cells_no=a.cells_no,
         save_cells_path=a.save_path)
