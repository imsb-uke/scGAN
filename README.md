# Realistic in silico generation and augmentation of single cell RNA-seq data using Generative Adversarial Neural Networks
This project contains the Tensorflow implementation and documentation for the training of the (c)scGAN models described in our [manuscript](https://www.nature.com/articles/s41467-019-14018-z) 'Realistic in silico generation and augmentation of single cell RNA-seq data using Generative Adversarial Neural Networks'.
This document describes how to set up an environment and run the code to replicate the results of the original manuscript.

## Main requirements
The code was tested with Python 3.5 and the packages listed in [requirements.txt](requirements.txt).
We assume that the installation of the above-mentioned packages covers all dependencies.
In case we have missed essential dependencies please raise an issue.
To allow you to reproduce our results easily, we also provide a Dockerfile that contains a working environment containing all the dependencies.
You can build the docker image using `docker build . -t scgan:latest -f dockerfile/Dockerfile` .
Please check the [Tensorflow Docker Help Page](https://www.tensorflow.org/install/docker?hl=en) for further information on how to use Tensorflow with docker.
We provide a ready-to-use Docker image on [DockerHub](https://hub.docker.com/r/fhausmann/scgan).

## Usage
The `main.py` script is used to start the pre-processing of the files, to start (or resume) the training, and generate cells using a trained model.
Model (hyper-) parameters can be adjusted in the 'parameters.json' file. 

### Parameter file
The (hyper-) parameters of the experiments are defined in a `parameters.json` file. A template is provided on this repository.
It contains the path to the directory where the processed data, models, logs and results will be stored, along with a list of GPU identifiers (0..n) to be used, given that each model will be trained on a single GPU.
For each experiment, a custom name (which will be used for the folder name) has to be provided.
When using the `main.py` script, the flag `--param` with the path to the `.json` parameter file should always be used.
Further details on the parameters can be found in the subsequent 'Data format / Pre-processing' and 'training' paragraphs.

### Data format / Pre-processing
Single cell RNA-seq data has to be in the `.h5` or `.h5ad` format. In case you want to use data in mtx/tsv, please use the provided Jupyter notebook to convert the data to `.h5` or `.h5ad`. For instance, most of the experiments reported in our paper use the [Fresh 68k PBMCs (Donor A) dataset](http://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz).
For performance issues, the data will be stored, after pre-processing and splitting, in TFRecord format, which in turn will be used for the training of the model.
All the parameters for the pre-processing of the data (filtering parameters, path to the dataset, size of the training / validation / test sets, clustering resolution) should be defined in the `parameters.json` parameter file.
Note that the validation set is used to produce the t-SNE plots (each embedding is computed using the whole validation set and an equal number of generated cells).
The test set is not used. Unless you want to use a test set for further custom experiments, you can set its size to 0.
To run the pre-processing, run the `main.py` script with the `--process` flag:
```sh
python main.py --param parameters.json --process
```
### Training
All the parameters regarding the training should be defined in the `.json` parameter file.
The type of model (scGAN or cscGAN), the size of the different layers, the normalization layer used to condition the generation (batchnorm or layernorm), lambda, the batch size, dimension of the latent space, along with all the optimizer parameters (number of steps, learning rate, algorithm used...) are defined there.
Also, the frequencies (how often the values are logged for the Tensorboard, the t-SNE validation plots are plotted, the loss values are displayed on the standard output, the model is saved) are defined there.
To run the training, the `--train` flagged is used.
```sh
python main.py --param parameters.json --train
```
You can use jointly the `--process` and `--train` flags in which case they will be run sequentially.

Note that you can also resume the training by specifying the path to a checkpoint in the `.json` file.

Here are how the t-SNE plots should look like (non-conditional on the left, conditional on the right)
![](/Misc/non-cond_t-SNE.jpg "non-conditional t-SNE") ![](/Misc/cond_t-SNE.jpg "conditional t-SNE")

### Tensorboard
After or during the training, you can use the Tensorboard to monitor the losses (for instance) evolve.
We have defined some summary operations on the most important quantities in the model.
Please refer to the [Tensorboard documentation](https://www.tensorflow.org/guide/summaries_and_tensorboard).
![](/Misc/Tensorboard.png "Tensorboard")

### Generation of cells
To generate cells and write them to a file, you can use the `--generate` flag. In that case, also use the `--cells_no` flag with a list of integers corresponding to the number of cells (one int per cluster index) you want to generate. The `--model_path` and `--save_path` flags also should be used, to specify the path to the model to be used and the path where to save the data respectively.
For instance:
```sh
python main.py --param parameters.json --generate --cells_no 1000 500 0 200 --model_path path/to/my/model --save_path where_to_save.h5ad
```