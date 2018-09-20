#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Functions that convert the expression data to TFRecords.
"""
import collections
import multiprocessing as mp
import os
import random
from os.path import join

import numpy as np
import tensorflow as tf

from preprocessing.process_raw import GeneMatrix

sct = collections.namedtuple('sc', ('barcode', 'count_no', 'genes_no', 'dset', 'cluster'))


def to_sparse(a):
    """
    Helper for conversion from dense to sparse matrix.

    Parameters
    ----------
    a : 2-D array
        A dense matrix.

    Returns
    -------
    A list of indices of the non-zero values.
    A list of the non-zero values.
    """
    flat = a.flatten()
    indices = np.nonzero(flat)
    values = flat[indices]
    return indices[0], values


def make_example(scg_line, barcode, count_no, genes_no,
                 cluster=None, categories=None):
    """
    Generates an Example from a line of a gene expression matrix + metadata

    Parameters
    ----------
    scg_line : 1-D numpy array
        A line from a gene expression matrix corresponding to a cell.
    barcode : str
        The barcode associated to the cell.
    count_no : int
        Number of total counts for the cell.
    genes_no : int
        Number of genes.
    cluster :
        Cluster index of the cell.
        Default is None.
    categories : list
        List of different cluster indices for one-hot encoding purposes.

    Returns
    -------
    An Example (tf.train API).

    """
    feat_map = {}

    idx, vals = to_sparse(scg_line)
    feat_map['indices'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=idx))

    feat_map['values'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=vals))

    feat_map['barcode'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[str.encode(barcode)]))

    feat_map['genes_no'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[genes_no]))

    feat_map['count_no'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[count_no]))

    # add hot encoding for classification problems
    if cluster:
        feat_map['cluster_1hot'] = tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=[int(c == cluster) for c in categories]))

        feat_map['cluster_int'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(cluster)]))

    return tf.train.Example(features=tf.train.Features(feature=feat_map))


def process_line(line):
    """
    Extracts a numpy 1-D array with expression levels and creates a sct
    (named tuple) from a line. If you want to apply postprocessing on the
    examples, this is the right place.

    Parameters
    ----------
    line : line from an AnnData object
        A line from the AnnData object associated to the dataset.

    Returns
    -------
    A numpy 1-D array with the cell expression levels.
    A named tuple with all the meta data.

    """
    dset = line.obs['split'][0]

    # metadata from line
    scmd = sct(barcode=line.obs_names[0],
               count_no=int(np.sum(line.X)),
               genes_no=line.obs['n_genes'][0],
               dset=dset,
               cluster=line.obs['cluster'][0])

    return line.X, scmd


def read_and_serialize(job_path):
    """
    Reads a raw file, and runs all the pipeline until creation of the TFRecords.

    Parameters
    ----------
    job_path : str
        Path to the job directory.

    Returns
    -------
    A string to be printed in the main process
    """

    sc_data = GeneMatrix(job_path)

    sc_data.apply_preprocessing()

    worker_path = join(job_path, 'TF_records')
    os.makedirs(worker_path, exist_ok=True)

    cat = sc_data.sc_raw.obs['cluster'].cat.categories
    with TFRFWriter(worker_path, categories=cat) as writer:
        for line in sc_data.sc_raw:
            sc_genes, d = process_line(line)

            # print in TF records
            writer.write_numpy(sc_genes, d.barcode, d.count_no,
                               d.genes_no, d.dset, d.cluster)

    return 'done with writing for: ' + job_path


def process_files(exp_folders):
    """
    Parallel pre-processing of the different experiments.

    Parameters
    ----------
    exp_folders : list of strings
        The path to the folder containing all the experiments.

    Returns
    -------

    """
    pool = mp.Pool()
    results = pool.imap_unordered(read_and_serialize, exp_folders)

    stat = []
    for res in results:
        print(res)
        stat.append(res)

    pool.close()
    pool.join()


class TFRFWriter:
    def __init__(self, out_dir, split_files=10, categories=None):
        """
        Default Constructor

        Parameters
        ----------
        out_dir : str
            Path to the directory where to write the TFRecords
        split_files : int
            Number of TFRecords files the dataset should be split into.
            Default is 10.
        categories : list
            List of the different cluster indices present in the dataset.
        """

        self.out_dir = out_dir

        self.train_filenames = [join(out_dir, 'train-%s.tfrecords' % i)
                                for i in range(split_files)]
        self.valid_filename = join(out_dir, 'validate.tfrecords')
        self.test_filename = join(out_dir, 'test.tfrecords')

        self.valid = None
        self.test = None
        self.train = None

        self.count = 0
        self.categories = categories
        self.count_train = 0
        self.count_eval = 0
        self.count_test = 0
        self.split_files = split_files

    def pick_file(self):
        """
        Randomly picks one of the TFRecord files

        Returns
        -------
        An int with the index of the TFRecord file.

        """
        return random.randint(0, self.split_files - 1)

    def write_numpy(self, sc_genes, barcode, count_no,
                    genes_no, dataset, cluster):
        """
        Writes a cell into a (randomly picked) TFRecord file.

        Parameters
        ----------
        sc_genes : array
            1-D array containing the expression levels of a cell
        barcode : str
            Barcode of that cell.
        count_no : int
            Total number of counts in that cell.
        genes_no : int
            Number of genes in that cell.
        dataset : str
            Identifies whether it's a training, validation or test cell.
        cluster : str
            Cluster index of the cell.

        Returns
        -------

        """

        example = make_example(sc_genes, barcode, count_no, genes_no, cluster,
                               categories=self.categories)
        strings = [example.SerializeToString()]
        self.write_tfrecords(strings, dataset)

    def write_tfrecords(self, records, dataset):
        """
        Writes the cell in the (randomly picked) TFRecord according to
        whether it's a training, validation or test cell.

        Parameters
        ----------
        records : list
            List of TFRecord objects.
        dataset : str
            Identifies whether it's a training, validation or test cell.

        Returns
        -------

        """

        cnt_inc = len(records)
        self.count += cnt_inc

        if dataset == 'test':
            self.count_test += cnt_inc
            for s in records:
                self.test.write(s)
        elif dataset == 'train':
            self.count_train += cnt_inc
            for s in records:
                self.train[self.pick_file()].write(s)
        elif dataset == 'valid':
            self.count_eval += cnt_inc
            for s in records:
                self.valid.write(s)
        else:
            raise ValueError("invalid dataset: %s" % dataset)

    def __enter__(self):
        """
        For context management.
        """
        self.open()
        return self

    def open(self):
        """
        Initializes the TFRecordWriter objects.

        Returns
        -------

        """
        opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        self.valid = tf.python_io.TFRecordWriter(self.valid_filename, opt)
        self.test = tf.python_io.TFRecordWriter(self.test_filename, opt)
        self.train = [tf.python_io.TFRecordWriter(filename, opt) for filename in
                      self.train_filenames]

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        For context management
        """
        self.close()

    def close(self):
        """
        Closes the files associated to the TFRecordWriter objects.

        Returns
        -------

        """

        try:
            self.test.close()
        except Exception as e:
            pass

        try:
            self.valid.close()
        except Exception as e:
            pass

        for f in self.train:
            try:
                f.close()
            except Exception as e:
                pass
