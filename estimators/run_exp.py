#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import json
from estimators.cscGAN import cscGAN
from estimators.scGAN import scGAN


def run_exp(exp_gpu, mode='train', cells_no=None, save_cells_path=None):
    """
    Function that runs the experiment.
    It loads the json parameter file, instantiates the correct model, runs the
     training or the generation of the cells.

    Parameters
    ----------
    exp_gpu : tuple
        Tuple containing first the path (string) to the experiment folder
         and second a list of available GPU indexes.
    mode : string
        If "train" is passed, the training will be started, else, it will
        generate cells using the model whose checkpoint is in the experiment
         folder (in a job sub-folder).
    cells_no : int or list
        Number of cells to generate.
        Should be a list with number per cluster for a cscGAN model.
        Default is None.
    save_cells_path : str
        Path in which the simulated cells should be saved.
        Default is None.

    Returns
    -------

    """

    # read the available GPU for training
    avail_gpus = exp_gpu[1]
    gpu_id = avail_gpus.pop(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)

    # read the parameters
    exp_folder = exp_gpu[0]
    with open(os.path.join(exp_folder, 'parameters.json')) as fp:
        hparams = json.load(fp)

    # find training and validation TF records
    input_tfr = os.path.join(exp_folder, 'TF_records')
    train_files = [os.path.join(input_tfr, f)
                   for f in os.listdir(input_tfr) if "train" in f]
    valid_files = [os.path.join(input_tfr, f)
                   for f in os.listdir(input_tfr) if "valid" in f]

    # log directory
    log_dir = os.path.join(exp_folder, 'job')

    if save_cells_path is None:
        save_cells_path = os.path.join(exp_folder, 'generated_cells.h5ad')

    tf.reset_default_graph()

    if hparams['model']['type'] == 'scGAN':

        gan_model = scGAN(
            train_files=train_files,
            valid_files=valid_files,
            genes_no=hparams['preprocessed']['genes_no'],
            scaling=hparams['preprocessed']['scale']['scaling'],
            scale_value=hparams['preprocessed']['scale']["scale_value"],
            max_steps=hparams['training']['max_steps'],
            batch_size=hparams['training']['batch_size'],
            latent_dim=hparams['model']['latent_dim'],
            gen_layers=hparams['model']['gen_layers'],
            output_lsn=hparams['model']['output_LSN'],
            critic_layers=hparams['model']['critic_layers'],
            optimizer=hparams['training']['optimizer']['algorithm'],
            lambd=hparams['model']['lambd'],
            beta1=hparams['training']['optimizer']['beta1'],
            beta2=hparams['training']['optimizer']['beta2'],
            decay=hparams['training']['learning_rate']['decay'],
            alpha_0=hparams['training']['learning_rate']['alpha_0'],
            alpha_final=hparams['training']['learning_rate']['alpha_final'])

        if mode == "train":
            gan_model.training(
                exp_folder=log_dir,
                valid_cells_no=hparams["preprocessed"]["valid_count"],
                checkpoint=hparams['training']['checkpoint'],
                progress_freq=hparams['training']['progress_freq'],
                validation_freq=hparams['training']['validation_freq'],
                critic_iter=hparams['training']['critic_iters'],
                summary_freq=hparams['training']['summary_freq'],
                save_freq=hparams['training']['save_freq'])

        else:
            gan_model.generate_cells(
                cells_no=int(cells_no),
                checkpoint=log_dir,
                save_path=save_cells_path)

    elif hparams['model']['type'] == 'cscGAN':

        gan_model = cscGAN(
            train_files=train_files,
            valid_files=valid_files,
            genes_no=hparams['preprocessed']['genes_no'],
            clusters_no=hparams['preprocessed']['clusters_no'],
            scaling=hparams['preprocessed']['scale']['scaling'],
            scale_value=hparams['preprocessed']['scale']["scale_value"],
            max_steps=hparams['training']['max_steps'],
            batch_size=hparams['training']['batch_size'],
            latent_dim=hparams['model']['latent_dim'],
            gen_layers=hparams['model']['gen_layers'],
            output_lsn=hparams['model']['output_LSN'],
            gene_cond_type=hparams['model']['gen_cond_type'],
            critic_layers=hparams['model']['critic_layers'],
            optimizer=hparams['training']['optimizer']['algorithm'],
            lambd=hparams['model']['lambd'],
            beta1=hparams['training']['optimizer']['beta1'],
            beta2=hparams['training']['optimizer']['beta2'],
            decay=hparams['training']['learning_rate']['decay'],
            alpha_0=hparams['training']['learning_rate']['alpha_0'],
            alpha_final=hparams['training']['learning_rate']['alpha_final'])

        if mode == "train":
            gan_model.training(
                exp_folder=log_dir,
                valid_cells_no=hparams["preprocessed"]["valid_count"],
                clusters_ratios=hparams['preprocessed']['clusters_ratios'],
                checkpoint=hparams['training']['checkpoint'],
                progress_freq=hparams['training']['progress_freq'],
                validation_freq=hparams['training']['validation_freq'],
                critic_iter=hparams['training']['critic_iters'],
                summary_freq=hparams['training']['summary_freq'],
                save_freq=hparams['training']['save_freq'])
        else:
            gan_model.generate_cells(
                cells_no=cells_no,
                checkpoint=log_dir,
                save_path=save_cells_path)

    avail_gpus.append(gpu_id)
