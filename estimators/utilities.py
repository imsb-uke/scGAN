#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
A number of classes (Critic and Generator) and helper functions used across all types of models.
"""
import tensorflow as tf
import numpy as np
import sys
import pandas as pd
import scanpy.api as sc
from estimators.AMSGrad import AMSGrad
from tensorflow.contrib.layers.python import layers
import scipy.sparse as sp_sparse
from natsort import natsorted
from MulticoreTSNE import MulticoreTSNE as TSNE
tsne = TSNE(n_jobs=8)


class Generator:
    """
    Generic class for the Generator network.
    """

    def __init__(self, fake_outputs, batch_size, latent_dim,
                 output_cells_dim, var_scope,  gen_layers,
                 output_lsn, gen_cond_type=None, is_training=None,
                 clusters_ratios=None, clusters_no=None,
                 input_clusters=None, reuse=None):
        """
        Default constructor.

        Parameters
        ----------
        fake_outputs : Tensor
            Tensor holding the output of the generator.
        batch_size : int
            Batch size used during the training.
        latent_dim : int
            Dimension of the latent space used from which the input noise
            of the generator is sampled.
        output_cells_dim : int
            Dimension of the output cells (i.e. the number of genes).
        var_scope : str
            Variable scope used for the created tensors.
        gen_layers : list
            List of integers corresponding to the number of neurons of each
            layer of the generator.
        output_lsn : int, None
            Parameter of the LSN layer at the output of the critic
            (i.e. total number of counts per
            generated cell).
            If set to None, the layer won't be added in the generator.
        gen_cond_type : str
            Conditional normalization layers used in the generator,
             can be either "batchnorm" or "layernorm".
            If anything else, it won't be added in the model ( there will be
            no conditioning in the generation).
            Default is None.
        is_training : Tensor
            Boolean placeholder encoding for whether we're in training or
            inference mode (for the batch normalization).
             Default is None.
        clusters_ratios : Tensor
            Placeholder containing the list of cluster ratios of the input data.
            Default is None.
        clusters_no : int
            Number of clusters.
            Default is None.
        input_clusters : Tensor
            Placeholders for the cluster indexes that should be used for
             conditional generation.
            Default is None.
        reuse : Boolean
            Whether to reuse the already existing Tensors.
            Default is None.
        """

        self.batch_size = batch_size
        self.is_training = is_training
        self.latent_dim = latent_dim
        self.output_cells_dim = output_cells_dim
        self.var_scope = var_scope
        self.gen_layers = gen_layers
        self.output_lsn = output_lsn
        self.gen_cond_type = gen_cond_type
        self.clusters_no = clusters_no
        self.input_clusters = input_clusters
        self.reuse = reuse
        self.fake_outputs = fake_outputs
        self.clusters_ratios = clusters_ratios

    @classmethod
    def create_cond_generator(cls, z_input, batch_size, latent_dim,
                              output_cells_dim, var_scope, gen_layers,
                              output_lsn, gen_cond_type, clusters_ratios,
                              is_training, clusters_no=None,
                              input_clusters=None, reuse=None):
        """
        Class method that instantiates a Generator and creates a
        conditional generator.

        Parameters
        ----------
        z_input : Tensor
            Tensor containing the noise used as input by the generator.
        batch_size : int
            Batch size used during the training.
        latent_dim : int
            Dimension of the latent space used from which the input noise
            of the generator is sampled.
        output_cells_dim : int
            Dimension of the output cells (i.e. the number of genes).
        var_scope : str
            Variable scope used for the created tensors.
        gen_layers : list
            List of integers corresponding to the number of neurons of
            each layer of the generator.
        output_lsn : int, None
            Parameter of the LSN layer at the output of the critic
            (i.e. total number of counts per generated cell).
        gen_cond_type : str
            conditional normalization layers used in the generator, can be
             either "batchnorm" or "layernorm". If anything else, it won't be
              added in the model (which means no conditional generation).
        clusters_ratios : Tensor
            Placeholder containing the list of cluster ratios of the input data.
        is_training : Tensor
            Boolean placeholder encoding for whether we're in training or
            inference mode (for the batch normalization).
        clusters_no : int
            Number of clusters.
            Default is None.
        input_clusters : Tensor
            Placeholders for the cluster indexes that should be used for
            conditional generation.
            Default is None.
        reuse : Boolean
            Whether to reuse the already existing Tensors.
            Default is None.

        Returns
        -------
        A conditional Generator object with the defined architecture.
        """

        with tf.variable_scope(var_scope, reuse=reuse):

            for i_lay, size in enumerate(gen_layers):
                with tf.variable_scope("generator_layers_" + str(i_lay + 1)):
                    z_input = layers.linear(
                        z_input,
                        size,
                        weights_initializer=layers.xavier_initializer(),
                        biases_initializer=None)

                    if i_lay != -1:
                        if gen_cond_type == "batchnorm":
                            z_input = batchnorm(
                                [0], z_input,
                                is_training=is_training,
                                labels=input_clusters,
                                n_labels=clusters_no)

                        elif gen_cond_type == "layernorm":
                            z_input = layernorm([1],
                                                z_input,
                                                labels=input_clusters,
                                                n_labels=clusters_no)

                    z_input = tf.nn.relu(z_input)

            with tf.variable_scope("generator_layers_" + 'output'):
                fake_outputs = layers.relu(
                    z_input, output_cells_dim,
                    weights_initializer=layers.variance_scaling_initializer(mode="FAN_AVG"),
                    biases_initializer=tf.zeros_initializer())

                if output_lsn is not None:
                    gammas_output = tf.Variable(
                        np.ones(z_input.shape.as_list()[0]) * output_lsn,
                        trainable=False)
                    sigmas = tf.reduce_sum(fake_outputs, axis=1)
                    scale_ls = tf.cast(gammas_output, dtype=tf.float32) / \
                        (sigmas + sys.float_info.epsilon)

                    fake_outputs = tf.transpose(tf.transpose(fake_outputs) *
                                                scale_ls)

            return cls(fake_outputs, batch_size, latent_dim, output_cells_dim,
                       var_scope, gen_layers, output_lsn,
                       gen_cond_type=gen_cond_type, is_training=is_training,
                       clusters_ratios=clusters_ratios, clusters_no=clusters_no,
                       input_clusters=input_clusters, reuse=reuse)

    @classmethod
    def create_generator(cls, z_input, batch_size, latent_dim,
                         output_cells_dim, var_scope, gen_layers,
                         output_lsn, is_training, reuse=None):
        """
        Class method that instantiates a Generator and creates a
         non-conditional generator.

        Parameters
        ----------
        z_input : Tensor
            Tensor containing the noise used as input by the generator.
        batch_size : int
            Batch size used during the training.
        latent_dim : int
            Dimension of the latent space used from which the input noise
             of the generator is sampled.
        output_cells_dim : int
            Dimension of the output cells (i.e. the number of genes).
        var_scope : str
            Variable scope used for the created tensors.
        gen_layers : list
            List of integers corresponding to the number of neurons of each
            layer of the generator.
        output_lsn : int, None
            Parameter of the LSN layer at the output of the critic
            (i.e. total number of counts per generated cell).
        is_training : Tensor
            Boolean placeholder encoding for whether we're in training or
             inference mode (for the  batch normalization).
        reuse : Boolean
            Whether to reuse the already existing Tensors.
            Default is None.

        Returns
        -------
        A Generator object with the defined architecture.
        """

        with tf.variable_scope(var_scope, reuse=reuse):

            for i_lay, size in enumerate(gen_layers):
                with tf.variable_scope("generator_layers_" + str(i_lay + 1)):
                    z_input = layers.linear(
                        z_input,
                        size,
                        weights_initializer=layers.xavier_initializer(),
                        biases_initializer=None
                        )
                    if i_lay != -1:
                        z_input = batchnorm([0], z_input, is_training)

                    z_input = tf.nn.relu(z_input)

            with tf.variable_scope("generator_layers_" + 'output'):
                fake_outputs = layers.relu(
                    z_input,
                    output_cells_dim,
                    weights_initializer=layers.variance_scaling_initializer(mode="FAN_AVG"),
                    biases_initializer=tf.zeros_initializer())

                if output_lsn is not None:
                    gammas_output = tf.Variable(
                        np.ones(z_input.shape.as_list()[0]) *
                        output_lsn, trainable=False)

                    sigmas = tf.reduce_sum(fake_outputs, axis=1)
                    scale_ls = tf.cast(gammas_output, dtype=tf.float32) /\
                        (sigmas + sys.float_info.epsilon)

                    fake_outputs = tf.transpose(tf.transpose(fake_outputs) *
                                                scale_ls)

        return cls(fake_outputs, batch_size, latent_dim, output_cells_dim, var_scope,
                   gen_layers, output_lsn, is_training=is_training, reuse=reuse)


class Critic:
    """
    Generic class for the Critic network
    """

    def __init__(self, xinput, dist, var_scope, critic_layers,
                 input_clusters=None, clusters_no=None, reuse=None):
        """
        Default constructor.

        Parameters
        ----------
        xinput : Tensor
            Tensor containing the input cells.
        dist : Tensor
            Tensor containing the output of the Critic
            (e.g. the Wasserstein distance).
        var_scope : str
            Variable scope used for the created tensors.
        critic_layers : list
            List of integers corresponding to the number of neurons of each
             layer of the critic.
        input_clusters : Tensor
            Tensor containing the corresponding cluster indexes of the input cells.
            Default is None.
        clusters_no : int
            Number of clusters.
            Default is None.
        reuse : Boolean
            Whether to reuse the already existing Tensors.
            Default is None.
        """

        self.xinput = xinput
        self.var_scope = var_scope
        self.critic_layers = critic_layers
        self.clusters_no = clusters_no
        self.input_clusters = input_clusters
        self.reuse = reuse
        self.dist = dist

    @classmethod
    def create_cond_critic(cls, xinput, input_clusters, var_scope,
                           critic_layers, clusters_no, reuse=None):
        """
        Class method that instantiates a Critic and creates a conditional critic.

        Parameters
        ----------
        xinput : Tensor
            Tensor containing the input cells.
        input_clusters : Tensor
            Tensor containing the corresponding cluster indexes of the input cells.
        var_scope : str
            Variable scope used for the created tensors.
        critic_layers : list
            List of integers corresponding to the number of neurons of each
            layer of the critic.
        clusters_no : int
            Number of clusters.
        reuse : Boolean
            Whether to reuse the already existing Tensors.
            Default is None.

        Returns
        -------
        A Creator object with the defined architecture.

        """

        with tf.variable_scope(var_scope, reuse=reuse):
            for i_lay, output_size in enumerate(critic_layers):
                with tf.variable_scope("layers_" + str(i_lay + 1)):
                    xinput = layers.relu(
                        xinput,
                        output_size,
                        weights_initializer=layers.variance_scaling_initializer(mode="FAN_AVG"),
                        biases_initializer=tf.zeros_initializer())

            with tf.variable_scope("layers_" + 'proj'):
                proj_weights_m = tf.get_variable(
                    "proj_weights_m",
                    [clusters_no, critic_layers[-1], 1],
                    dtype=tf.float32,
                    initializer=layers.xavier_initializer())

                proj_bias_m = tf.get_variable(
                    "proj_bias_m",
                    [clusters_no, 1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer())

                proj_weights = tf.nn.embedding_lookup(
                    proj_weights_m, input_clusters)
                proj_bias = tf.nn.embedding_lookup(proj_bias_m, input_clusters)

                output = tf.einsum('ij,ijk->ik', xinput, proj_weights)
                dist = tf.add(output, proj_bias)

        return cls(xinput, dist, var_scope, critic_layers,
                   input_clusters=input_clusters,
                   clusters_no=clusters_no, reuse=reuse)

    @classmethod
    def create_cond_critic_proj(cls, xinput, input_clusters, var_scope,
                           critic_layers, clusters_no, reuse=None):
        """
        Class method that instantiates a Critic and creates a conditional
         critic with the original projection conditioning method.

        Parameters
        ----------
        xinput : Tensor
            Tensor containing the input cells.
        input_clusters : Tensor
            Tensor containing the corresponding cluster indexes of the input cells.
        var_scope : str
            Variable scope used for the created tensors.
        critic_layers : list
            List of integers corresponding to the number of neurons of each
             layer of the critic.
        clusters_no : int
            Number of clusters.
        reuse : Boolean
            Whether to reuse the already existing Tensors.
            Default is None.

        Returns
        -------
        A Creator object with the defined architecture.

        """

        with tf.variable_scope(var_scope, reuse=reuse):
            for i_lay, output_size in enumerate(critic_layers):
                with tf.variable_scope("layers_" + str(i_lay + 1)):
                    xinput = layers.relu(
                        xinput,
                        output_size,
                        weights_initializer=layers.variance_scaling_initializer(mode="FAN_AVG"),
                        biases_initializer=tf.zeros_initializer())

            with tf.variable_scope("layers_" + 'proj'):
                proj_weights_m = tf.get_variable(
                    "proj_weights_m",
                    [clusters_no, critic_layers[-1], 1],
                    dtype=tf.float32, initializer=layers.xavier_initializer())

                proj_weights = tf.nn.embedding_lookup(proj_weights_m,
                                                      input_clusters)

                output_proj = tf.einsum('ij,ijk->ik', xinput, proj_weights)

            with tf.variable_scope("layers_" + 'output'):
                output = layers.linear(
                    xinput, 1,
                    weights_initializer=layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer())

                dist = tf.add(output_proj, output)

        return cls(xinput, dist, var_scope, critic_layers,
                   input_clusters=input_clusters,
                   clusters_no=clusters_no, reuse=reuse)

    @classmethod
    def create_critic(cls, xinput, var_scope,
                      critic_layers, reuse=None):
        """
        Class method that instantiates a Critic and creates a
         non-conditional critic.

        Parameters
        ----------
        xinput : Tensor
            Tensor containing the input cells.
        var_scope : str
            Variable scope used for the created tensors.
        critic_layers : list
            List of integers corresponding to the number of neurons of each
             layer of the critic.
        reuse : Boolean
            Whether to reuse the already existing Tensors.
            Default is None.

        Returns
        -------
        A Creator object with the defined architecture.

        """

        with tf.variable_scope(var_scope, reuse=reuse):
            for i_lay, output_size in enumerate(critic_layers):
                with tf.variable_scope("layers_" + str(i_lay + 1)):
                    xinput = layers.relu(
                        xinput, output_size,
                        weights_initializer=layers.variance_scaling_initializer(mode="FAN_AVG"),
                        biases_initializer=tf.zeros_initializer())

            with tf.variable_scope("layers_" + 'output'):
                output = layers.linear(
                    xinput, 1,
                    weights_initializer=layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer())

        return cls(xinput, output, var_scope, critic_layers, reuse=reuse)


def add_random_labels(clusters_ratios, batch_size):
    """
    Function that randomly samples cluster indices following a
    multinomial distribution.

    Parameters
    ----------
    clusters_ratios : Tensor
        Placeholder containing the parameters of the multinomial distribution.
    batch_size : int
        Batch size used during the training.

    Returns
    -------
    A Tensor containing a batch of sampled indices.

    """
    clusters_ratios = tf.cast(clusters_ratios, dtype=tf.float32)

    mn_logits = tf.tile(tf.log(clusters_ratios),
                        (batch_size, 1))

    labels = tf.to_int32(tf.multinomial(mn_logits, 1))

    return tf.squeeze(labels)


def add_random_input(batch_size, latent_dim):
    """
    Function that randomly samples from the latent space distribution
    (e.g. a standard Gaussian).

    Parameters
    ----------
    batch_size : int
        Batch size used during the training.
    latent_dim : int
        Dimension of the Gaussian (or latent space) to sample from.

    Returns
    -------
    A Tensor containing a batch of sampled inputs.

    """
    return tf.random_normal([batch_size, latent_dim])


def save_generated_cells(fake_cells, file_name, fake_labels=None):
    """
    Functions that writes a gene expression matrix and the associated
    cluster indices into a file. Check the AnnData documentation of the
     write method to check the supported formats.

    Parameters
    ----------
    fake_cells : 2-D array
        A matrix (cells x genes) containing the expression levels.
        It can be dense or sparse. It will be encoded in a sparse format.
    file_name : str
        Path of the file to write to.
    fake_labels : array
        an array containing the cluster indices of the corresponding cells.
        Default is None.

    Returns
    -------

    """

    s_gen_mat = sp_sparse.csr_matrix(fake_cells)
    sc_fake = sc.AnnData(s_gen_mat)

    if fake_labels is not None:
        groups = fake_labels.astype('U')
        unique_groups = np.unique(groups)
        sc_fake.obs['cluster'] = pd.Categorical(
            values=groups,
            categories=natsorted(unique_groups))

    sc_fake.obs_names = np.repeat('fake', sc_fake.shape[0])
    sc_fake.obs_names_make_unique()

    sc_fake.write(file_name)


def rescale(fake_cells, scaling, scale_value):
    """
    Function to "scale back" the scRNAseq data.
    Currently, only Library Size Normalization is supported.

    Parameters
    ----------
    fake_cells : 2-D array
        A matrix (cells x genes) containing the expression levels.
    scaling : str
        Method used to scale.
        Check the code directly.
    scale_value : int, float or list
        Parameter of the scaling function.

    Returns
    -------
    The scaled back expression matrix.

    """

    if "normalize_per_cell_LS_" in str(scaling):
        fake_cells = fake_cells * float(scale_value)

    return fake_cells


def sc_summary(name_scope, sc_batch):
    """
    Creates mean and histogram of some batch quantities for the Tensorboard.

    Parameters
    ----------
    name_scope : str
        Name scope used to visualize in the Tensorboard.
    sc_batch : Tensor
        Tensor containing the batch of the quantity to summarize.

    Returns
    -------

    """
    with tf.name_scope(name_scope):
        mean = tf.reduce_mean(sc_batch)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', sc_batch)


def set_learning_rate(alpha_0, alpha_final, global_step, max_steps):
    """
    Creates a (exponentially) decaying learning rate tensor to be used
    with the optimizer.

    Parameters
    ----------
    alpha_0 : float
        initial learning rate
    alpha_final : float
        final learning rate
    global_step : Tensor
        Contains the index of the current global step of the learning procedure.
    max_steps : int
        Total number of steps used in the learning phase.

    Returns
    -------
    A Tensor with the decaying rate.

    """
    learning_rate = tf.train.exponential_decay(
        learning_rate=alpha_0,
        global_step=global_step,
        decay_steps=max_steps,
        decay_rate=alpha_final / alpha_0)

    return learning_rate


def set_global_step():
    """
    Function that creates an operation to increment the global step
     (and instantiates it if necessary).

    Returns
    -------
    A Tensor containing the current training step (used for the optimizer).
    A Operation that increments the global step (to be called once in each outer
     iteration of the learning algorithm).
    """
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)
    return global_step, incr_global_step


def gradient_step(training_variables, training_loss, learning_rate,
                  beta1, beta2, optimizer='AMSGrad'):
    """
    Function that creates ops that compute and apply a gradient descent step.

    Parameters
    ----------
    training_variables : Variable or Collection
        Training variables for which the gradients are computed and applied.
    training_loss : Tensor
        Contains the loss function to be optimized.
    learning_rate : Tensor
        Contains the (potentially exponentially decreasing) leaning rate.
    beta1 : float
        Exponential decay for the first-moment estimates.
    beta2 : float
        Exponential decay for the second-moment estimates.
    optimizer : str
        Optimizer to be used.
        Can be "AMSGrad" for AMSGrad optimizer.
        Else, uses Adam.
        Default is "AMSGrad".

    Returns
    -------
    An operation to apply the gradient step.
    A Tensor containing the gradients for all specified variables.

    """

    if optimizer == 'AMSGrad':
        optimizer = AMSGrad(learning_rate=learning_rate,
                            beta1=beta1,
                            beta2=beta2)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1,
                                           beta2=beta2)

    grads_and_vars = optimizer.compute_gradients(training_loss,
                                                 var_list=training_variables)

    return optimizer.apply_gradients(grads_and_vars), grads_and_vars


def batchnorm(axes, inputs, is_training, decay=0.999,
              labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016)"""

    moving_mean = tf.get_variable("BN_moving_mean",
                                  inputs.get_shape().as_list()[1:],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer(),
                                  trainable=False)

    moving_var = tf.get_variable("BN_moving_var",
                                 inputs.get_shape().as_list()[1:],
                                 dtype=tf.float32,
                                 initializer=tf.ones_initializer(),
                                 trainable=False)

    n_neurons = inputs.get_shape().as_list()[1]

    if labels is None:
        with tf.variable_scope("BLN"):
            offset = tf.get_variable("BN_offset",
                                     [n_neurons],
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
            scale = tf.get_variable("BN_scale",
                                    [n_neurons],
                                    dtype=tf.float32,
                                    initializer=tf.ones_initializer())
    else:
        with tf.variable_scope("BLN"):
            offset_m = tf.get_variable("BN_offset",
                                       [n_labels] + inputs.get_shape().as_list()[1:],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())

            scale_m = tf.get_variable("BN_scale",
                                      [n_labels] + inputs.get_shape().as_list()[1:],
                                      dtype=tf.float32,
                                      initializer=tf.ones_initializer())

        offset = tf.squeeze(tf.nn.embedding_lookup(offset_m, labels))
        scale = tf.squeeze(tf.nn.embedding_lookup(scale_m, labels))

    def bn_training():
        batch_mean, batch_var = tf.nn.moments(inputs, axes, keep_dims=False)

        upd_moving_mean = tf.assign(
            moving_mean, moving_mean * decay + batch_mean * (1 - decay))
        upd_moving_var = tf.assign(
            moving_var, moving_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([upd_moving_mean, upd_moving_var]):
            return tf.nn.batch_normalization(
                inputs, batch_mean, batch_var, offset, scale, 1e-5)

    def bn_inference():
        return tf.nn.batch_normalization(
            inputs, moving_mean, moving_var, offset, scale, 1e-5)

    return tf.cond(is_training, bn_training, bn_inference)


def layernorm(axes, inputs, labels=None, n_labels=None):
    """conditional layernorm, inspired by batchnorm (dumoulin et al 2016)"""

    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)

    # Assume the 'neurons' axis is the first of norm_axes.
    # This is the case for fully-connected and BCHW conv layers.
    n_neurons = inputs.get_shape().as_list()[axes[0]]

    if labels is None:
        with tf.variable_scope("BLN"):
            offset = tf.get_variable("LN_offset",
                                     [n_neurons],
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
            scale = tf.get_variable("LN_scale",
                                    [n_neurons],
                                    dtype=tf.float32,
                                    initializer=tf.ones_initializer())

    else:
        with tf.variable_scope("BLN"):
            offset_m = tf.get_variable("LN_offset",
                                       [n_labels, n_neurons],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            scale_m = tf.get_variable("LN_scale",
                                      [n_labels, n_neurons],
                                      dtype=tf.float32,
                                      initializer=tf.ones_initializer())

        offset = tf.squeeze(tf.nn.embedding_lookup(offset_m, labels))
        scale = tf.squeeze(tf.nn.embedding_lookup(scale_m, labels))

    return tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)
