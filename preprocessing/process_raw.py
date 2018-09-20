#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import ntpath
import os
import random
import warnings
from collections import Counter, namedtuple
import numpy as np
import pandas as pd
import scanpy.api as sc
import scipy.sparse as sp_sparse
from natsort import natsorted

SEED = 0


class GeneMatrix:
    """
    Class used to represent the scRNAseq data. Handles reading the data from a
    given path, the pre-processing and writing to a given path.
    """
    def __init__(self, job_path):
        """
        Default constructor.

        Parameters
        ----------
        job_path : str
            Path to the job directory containing the parameters.json file
        """

        self.job_dir = job_path

        self.train_cells = None
        self.valid_cells = None
        self.test_cells = None
        self.balanced_split = None
        self.split_seed = None
        self.cluster_res = None
        self.min_genes = 0
        self.min_cells = 0
        self.scale = None
        self.raw_file = None
        self.parse_paramters()

        self.sc_raw = None
        self.read_raw_file()

        self.train_cells_per_cluster = None
        self.valid_cells_per_cluster = None
        self.test_cells_per_cluster = None

        self.clusters_no = None
        self.clusters_ratios = None

        self.cells_count = None
        self.genes_count = None

    def parse_paramters(self):
        """
        Method that parses the parameters.json file and populates
         the attributes of the object accordingly.

        Returns
        -------

        """

        with open(os.path.join(self.job_dir, 'parameters.json'), 'r') as fp:
            hparam = json.load(fp)

        self.raw_file = os.path.join(
            self.job_dir, ntpath.basename(hparam['input_ds']['raw_input']))

        self.valid_cells = hparam['input_ds']['split']['valid_cells']
        self.test_cells = hparam['input_ds']['split']['test_cells']

        self.balanced_split = hparam['input_ds']['split']['balanced_split']
        self.split_seed = hparam['input_ds']['split']['split_seed']

        self.cluster_res = hparam['input_ds']['clustering']['res']

        self.min_genes = hparam['input_ds']['filtering']['min_genes']
        self.min_cells = hparam['input_ds']['filtering']['min_cells']

        self.scale = hparam['input_ds']['scale']

    def read_raw_file(self):
        """
        Reads the raw data file and turns it into a dense matrix, stored as
        an attribute.

        Returns
        -------

        """

        print("reading single cell data from {}".format(self.raw_file))

        file_format = self.raw_file.split('.')[-1]

        if file_format == 'h5':
            andata = sc.read_10x_h5(self.raw_file)
        elif file_format == 'h5ad':
            andata = sc.read(self.raw_file)
        else:
            raise ValueError('Reading [ %s ] failed, the inferred file '
                             'format [ %s ] is not supported. Please convert '
                             'your file to either h5 or h5ad format.'
                             % (self.raw_file, file_format))

        # appends -1 -2... to the name of genes that already exist
        andata.var_names_make_unique()
        if sp_sparse.issparse(andata.X):
            andata.X = andata.X.toarray()

        self.sc_raw = andata

    def clustering(self):
        """
        Method that applies a Louvain clustering of the data, following
        the Zheng recipe. Computes and stores the cluster ratios.

        Returns
        -------

        """

        if self.cluster_res is None:
            if "cluster" in self.sc_raw.obs_keys():
                print("clustering is already done,"
                      " no clustering will be applied")
            else:
                raise ValueError(' No clustering is applied, '
                                 'please apply clustering')
        else:
            clustered = self.sc_raw.copy()

            # pre-processing
            sc.pp.recipe_zheng17(clustered)
            sc.tl.pca(clustered, n_comps=50)

            # clustering
            sc.pp.neighbors(clustered, n_pcs=50)
            sc.tl.louvain(clustered, resolution=self.cluster_res)

            # add clusters to the raw data
            self.sc_raw.obs['cluster'] = clustered.obs['louvain']

        # adding clusters' ratios
        cells_per_cluster = Counter(self.sc_raw.obs['cluster'])
        clust_ratios = dict()
        for key, value in cells_per_cluster.items():
            clust_ratios[key] = value / self.sc_raw.shape[0]

        self.clusters_ratios = clust_ratios
        self.clusters_no = len(cells_per_cluster)
        print("Clustering of the raw data is done to %d clusters."
              % self.clusters_no)

    def filtering(self):
        """
        Filters the data (discarding genes expressed in too few cells
         and cells that express too few genes, according to the parameters
         given in the json file.

        Returns
        -------

        """

        sc.pp.filter_cells(self.sc_raw, min_genes=self.min_genes, copy=False)
        print("Filtering of the raw data is done with minimum "
              "%d cells per gene." % self.min_genes)

        sc.pp.filter_genes(self.sc_raw, min_cells=self.min_cells, copy=False)
        print("Filtering of the raw data is done with minimum"
              " %d genes per cell." % self.min_cells)

        self.cells_count = self.sc_raw.shape[0]
        self.genes_count = self.sc_raw.shape[1]
        print("Cells number is %d , with %d genes per cell."
              % (self.cells_count, self.genes_count))

    def scaling(self):
        """
        Method that scales the expression data.
        Currently only supports Library Size Normalization.

        Returns
        -------

        """

        if "normalize_per_cell_LS_" in str(self.scale):

            lib_size = int(self.scale.split('_')[-1])
            sc.pp.normalize_per_cell(self.sc_raw,
                                     counts_per_cell_after=lib_size)
            self.scale = {"scaling": 'normalize_per_cell_LS',
                          "scale_value": lib_size}

        else:

            warnings.warn("The scaling of the data is unknown, library size "
                          "library size normalization with 20k will be applied")

            lib_size = int(self.scale.split('_')[-1])
            sc.pp.normalize_per_cell(self.sc_raw,
                                     counts_per_cell_after=lib_size)
            self.scale = {"scaling": 'normalize_per_cell_LS',
                          "scale_value": lib_size}

        print("Scaling of the data is done using " + self.scale["scaling"]
              + "with " + str(self.scale["scale_value"]))

    def split(self):
        """
        Splits the data into training, validation and test sets, using the
         ratios defined in the json file. Supports "balanced splitting"
          to ensure the cluster ratios are respected in each split.

        Returns
        -------

        """

        # one seed to be used for all experiments
        if self.split_seed == 'default':
            self.split_seed = SEED

        random.seed(self.split_seed)
        np.random.seed(self.split_seed)

        if self.balanced_split:

            valid_cells_per_cluster = {
                key: int(value * self.valid_cells)
                for key, value in self.clusters_ratios.items()}

            test_cells_per_cluster = {
                key: int(value * self.test_cells)
                for key, value in self.clusters_ratios.items()}

            dataset = np.repeat('train', self.sc_raw.shape[0])
            unique_groups = np.asarray(['valid', 'test', 'train'])
            self.sc_raw.obs['split'] = pd.Categorical(
                values=dataset,
                categories=natsorted(unique_groups))

            for key in valid_cells_per_cluster:

                # all cells from clus idx
                indices = self.sc_raw.obs[
                    self.sc_raw.obs['cluster'] == str(key)].index

                test_valid_indices = np.random.choice(
                    indices, valid_cells_per_cluster[key] +
                    test_cells_per_cluster[key], replace=False)

                test_indices = test_valid_indices[0:test_cells_per_cluster[key]]
                valid_indices = test_valid_indices[test_cells_per_cluster[key]:]

                for i in test_indices:
                    self.sc_raw.obs.set_value(i, 'split', 'test')

                for i in valid_indices:
                    self.sc_raw.obs.set_value(i, 'split', 'valid')

            self.valid_cells_per_cluster = valid_cells_per_cluster
            self.test_cells_per_cluster = test_cells_per_cluster

        else:

            dataset = np.repeat('train', self.sc_raw.shape[0])

            unique_groups = np.asarray(['valid', 'test', 'train'])

            self.sc_raw.obs['split'] = pd.Categorical(
                values=dataset,
                categories=natsorted(unique_groups))

            # all cells from clus idx
            indices = self.sc_raw.obs.index

            test_valid_indices = np.random.choice(
                indices,
                self.test_cells + self.valid_cells,
                replace=False)

            test_indices = test_valid_indices[0:self.test_cells]
            valid_indices = test_valid_indices[self.test_cells:]

            for i in test_indices:
                self.sc_raw.obs.set_value(i, 'split', 'test')

            for i in valid_indices:
                self.sc_raw.obs.set_value(i, 'split', 'valid')

            self.valid_cells_per_cluster = Counter(
                self.sc_raw[valid_indices].obs['cluster'])

            self.test_cells_per_cluster = Counter(
                self.sc_raw[test_indices].obs['cluster'])

        train_indices = self.sc_raw[
            self.sc_raw.obs['split'] == 'train'].obs.index

        self.train_cells_per_cluster = dict(
            Counter(self.sc_raw[train_indices].obs['cluster']))

        self.train_cells = self.cells_count - self.test_cells - self.valid_cells

    def write(self):
        """
        Writes a data file in the job directory after filtering, scaling,
         clustering, splitting...Also writes an extended, specfic json file,
          for the corresponding experiment.

        Returns
        -------

        """

        # write the single cell clustered and processed file
        self.sc_raw.write(self.raw_file)

        with open(os.path.join(self.job_dir, 'parameters.json'), 'r') as fp:
            hparam = json.load(fp)

        # dump json param in this dir
        hparam['preprocessed'] = {
            'total_count': self.cells_count,
            'genes_no': self.genes_count,
            'train_count': self.train_cells,
            'valid_count': self.valid_cells,
            'test_count': self.test_cells,
            'train_cells_per_cluster': self.train_cells_per_cluster,
            'valid_cells_per_cluster': self.valid_cells_per_cluster,
            'test_cells_per_cluster': self.test_cells_per_cluster,
            'split_seed': self.split_seed,
            'scale': self.scale,
            'clusters_no': self.clusters_no,
            'clusters_ratios': self.clusters_ratios}

        with open(os.path.join(self.job_dir, 'parameters.json'), 'w') as fp:
            fp.write(json.dumps(hparam, sort_keys=True, indent=4))

    def apply_preprocessing(self):
        """
        Executes all the preprocessing steps on the GeneMatrix class :
        reading the file, clustering, filtering, scaling, splitting and writing
         the output to a classic (h5 or h5ad) file.

        Returns
        -------

        """
        # apply clustering when needed
        self.clustering()

        # apply basic global filtering and scaling
        self.filtering()

        # apply basic global filtering and scaling
        self.scaling()

        # apply balanced split to test train and validation, when required
        self.split()

        # update the raw file
        self.write()
