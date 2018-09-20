#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from estimators.utilities import Generator, Critic
from estimators.utilities import add_random_input, add_random_labels
from estimators.utilities import set_learning_rate, set_global_step, rescale
from estimators.utilities import gradient_step, sc_summary, save_generated_cells
from MulticoreTSNE import MulticoreTSNE as TSNE
tsne = TSNE(n_jobs=20)


class cscGAN:
    """
    Contains the class for the conditional scGAN (cscGAN).
    Methods include the creation of the graph, the training of the model,
     the validation and generation of the cells.
    """
    def __init__(self, train_files, valid_files, genes_no, clusters_no,
                 scaling, scale_value, max_steps, batch_size, latent_dim,
                 gen_layers, output_lsn, gene_cond_type, critic_layers,
                 optimizer, lambd, beta1, beta2, decay, alpha_0, alpha_final):
        """
        Constructor for the cscGAN.

        Parameters
        ----------
        train_files : list
            List of TFRecord files used for training.
        valid_files : list
            List of TFRecord files used for validation.
        genes_no : int
            Number of genes in the expression matrix.
        clusters_no : int
            Number of clusters.
        scaling : str
            Method used to scale the data, see the scaling method of the
            GeneMatrix class in preprocessing/process_raw.py for more details.
        scale_value : int, float
            Parameter of the scaling function.
        max_steps : int
            Number of steps in the (outer) training loop.
        batch_size : int
            Batch size used for the training.
        latent_dim : int
            Dimension of the latent space used from which the input noise
             of the generator is sampled.
        gen_layers : list
            List of integers corresponding to the number of neurons of each
            layer of the generator.
        output_lsn : int, None
            Parameter of the LSN layer at the output of the critic
             (i.e. total number of counts per generated cell).
            If set to None, the layer won't be added in the generator.
        gene_cond_type : str
            Conditional normalization layers used in the generator.
             Can be either "batchnorm" or "layernorm".
            If anything else, it won't be added in the model
            (there will be no conditioning in the generation).
        critic_layers : list
            List of integers corresponding to the number of neurons of each
            layer of the critic.
        optimizer : str
            Optimizer used in the training.
            Can be "AMSGrad" for AMSGrad.
            If anything else, Adam will be used.
        lambd : float
            Regularization hyper-parameter to be used with the gradient
             penalty in the WGAN loss.
        beta1 : float
            Exponential decay for the first-moment estimates.
        beta2 : float
            Exponential decay for the second-moment estimates.
        decay : str
            If True, uses an exponential decay of the learning rate.
        alpha_0 : float
            Initial learning rate value.
        alpha_final : float
            Final value of the learning rate if the decay is set to True.
        """

        # read the parameters
        self.clusters_no = clusters_no
        self.latent_dim = latent_dim
        self.lambd = lambd
        self.gen_cond_type = gene_cond_type
        self.critic_layers = critic_layers
        self.gen_layers = gen_layers
        self.output_lsn = output_lsn
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_decay = decay
        self.alpha_0 = alpha_0
        self.alpha_final = alpha_final
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.scaling = scaling
        self.scale_value = scale_value
        self.train_files = train_files
        self.valid_files = valid_files
        self.genes_no = genes_no

        # prepare input pipeline for training
        self.train_cells, self.train_cells_clusters = self.make_input_fn(
            self.train_files)

        # prepare input pipeline for validation
        self.test_cells, self.test_cells_clusters = self.make_input_fn(
            self.valid_files)

        # module parameters
        self.generator = None
        self.critic_real = None
        self.critic_fake = None
        self.gradient_penalty = None
        self.gen_loss = None
        self.critic_loss = None
        self.global_step = None
        self.incr_global_step = None
        self.learning_rate = None
        self.critic_train = None
        self.critic_grads_and_vars = None
        self.gen_train = None
        self.gen_grads_and_vars = None
        self.model_train = None
        self.output_tensor = None
        self.build_model()

        # add visualization
        self.visualization()

        # the total number of all trainable parameters
        self.parameter_count = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    def make_input_fn(self, file_paths, epochs=None):
        """
        Function that loads the TFRecords files and creates the placeholders
        for the data inputs.

        Parameters
        ----------
        file_paths : list
            List of TFRecord files from which to read from.
        epochs : int
            Integer specifying the number of times to read through the dataset.
            If None, cycles through the dataset forever.
            NOTE - If specified, creates a variable that must be initialized,
            so call tf.local_variables_initializer() and run the op in a session.
            Default is None.

        Returns
        -------
        features : Tensor
            Tensor containing a batch of cells (vector of expression levels).
        cluster : Tensor
            Tensor containing (a batch of) the cluster indexes of the
            corresponding cells.
        """

        feature_map = {'scg': tf.SparseFeature(index_key='indices',
                                               value_key='values',
                                               dtype=tf.float32,
                                               size=self.genes_no),
                       'cluster_int': tf.FixedLenFeature(1, tf.int64)}

        options = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.GZIP)

        batched_features = tf.contrib.learn.read_batch_features(
            file_pattern=file_paths,
            batch_size=self.batch_size,
            features=feature_map,
            reader=lambda: tf.TFRecordReader(
                options=options),
            num_epochs=epochs)

        sgc = batched_features['scg']

        sparse = tf.sparse_reshape(sgc, (self.batch_size, self.genes_no))

        dense = tf.sparse_tensor_to_dense(sparse)

        cluster = tf.squeeze(tf.to_int32(batched_features['cluster_int']))

        features = tf.reshape(dense, (self.batch_size, self.genes_no))

        return features, cluster

    def build_model(self):
        """
        Method that initializes the cscGAN model, creates the graph and
        defines the loss and optimizer.

        Returns
        -------

        """
        # training or inference (used for the batch normalization)
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        clusters_ratios = tf.placeholder(dtype=tf.float32,
                                         shape=(1, self.clusters_no),
                                         name='clusters_ratios')

        z_input = add_random_input(self.batch_size, self.latent_dim)
        input_clusters = add_random_labels(clusters_ratios, self.batch_size)

        # create generator
        self.generator = Generator.create_cond_generator(
            z_input=z_input,
            batch_size=self.batch_size,
            latent_dim=self.latent_dim,
            output_cells_dim=self.genes_no,
            var_scope='generator',
            gen_layers=self.gen_layers,
            output_lsn=self.output_lsn,
            gen_cond_type=self.gen_cond_type,
            clusters_no=self.clusters_no,
            input_clusters=input_clusters,
            is_training=is_training,
            clusters_ratios=clusters_ratios,
            reuse=None)

        # Critic with real cells as input
        with tf.name_scope('real_critic'):
            self.critic_real = Critic.create_cond_critic(
                xinput=self.train_cells,
                input_clusters=self.train_cells_clusters,
                var_scope="critic",
                critic_layers=self.critic_layers,
                clusters_no=self.clusters_no,
                reuse=None)

        # Critic with generated cells as input (shares weights with critic_real)
        with tf.name_scope('fake_critic'):
            self.critic_fake = \
                Critic.create_cond_critic(
                    xinput=self.generator.fake_outputs,
                    input_clusters=self.generator.input_clusters,
                    var_scope="critic",
                    critic_layers=self.critic_layers,
                    clusters_no=self.clusters_no, reuse=True)

        # Disc loss
        with tf.name_scope('critic_loss'):
            critic_loss_wgan = tf.reduce_mean(self.critic_fake.dist) \
                               - tf.reduce_mean(self.critic_real.dist)

            # The following lines implement the gradient penalty term
            alpha = tf.random_uniform(shape=[self.batch_size, 1],
                                      minval=0., maxval=1.)

            generator_interpolates = \
                Generator.create_cond_generator(
                    z_input=z_input,
                    batch_size=self.batch_size,
                    latent_dim=self.latent_dim,
                    output_cells_dim=self.genes_no,
                    var_scope='generator',
                    gen_layers=self.gen_layers,
                    output_lsn=self.output_lsn,
                    gen_cond_type=self.gen_cond_type,
                    clusters_no=self.clusters_no,
                    input_clusters=self.train_cells_clusters,
                    is_training=is_training,
                    clusters_ratios=clusters_ratios,
                    reuse=True)

            differences = generator_interpolates.fake_outputs - self.train_cells

            interpolates = self.train_cells + (alpha * differences)

            with tf.name_scope('help_critic'):
                critic_interpolates = \
                    Critic.create_cond_critic(
                        xinput=interpolates,
                        input_clusters=self.train_cells_clusters,
                        var_scope="critic",
                        critic_layers=self.critic_layers,
                        clusters_no=self.clusters_no,
                        reuse=True)

            gradients = tf.gradients(critic_interpolates.dist, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                           reduction_indices=[1]))
            self.gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
            critic_loss_wgan += self.lambd * self.gradient_penalty
            self.critic_loss = critic_loss_wgan

        # gen loss
        with tf.name_scope('generator_loss'):
            self.gen_loss = -tf.reduce_mean(self.critic_fake.dist)

        # add global step
        self.global_step, self.incr_global_step = set_global_step()

        # add decaying learning rate
        self.learning_rate = set_learning_rate(self.alpha_0,
                                               self.alpha_final,
                                               self.global_step,
                                               self.max_steps)

        # training the critic
        with tf.name_scope("critic_train"):
            critic_params = [var for var in tf.trainable_variables()
                             if var.name.startswith('critic')]
            self.critic_train, self.critic_grads_and_vars =\
                gradient_step(critic_params,
                              training_loss=self.critic_loss,
                              learning_rate=self.learning_rate,
                              beta1=self.beta1,
                              beta2=self.beta2,
                              optimizer=self.optimizer)

        # training the generator
        with tf.name_scope("generator_train"):
            with tf.control_dependencies(
                    [self.critic_train] +
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                gen_params = [var for var in tf.trainable_variables()
                              if var.name.startswith('gen')]
                self.gen_train, self.gen_grads_and_vars = gradient_step(
                    gen_params,
                    training_loss=self.gen_loss,
                    learning_rate=self.learning_rate,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    optimizer=self.optimizer)

        self.model_train = tf.group(self.incr_global_step, self.gen_train)
        self.critic_train = tf.group(self.critic_train)

    def visualization(self):
        """
        Method creating the placeholders to log the values for monitoring
        with the Tensorboard.

        Returns
        -------

        """

        # histograms and mean of real and generated cells
        sc_summary("single_cell_fake", self.generator.fake_outputs)
        sc_summary("single_cell_real", self.train_cells)

        # loss functions visualization
        tf.summary.scalar("gen_loss", self.gen_loss)
        tf.summary.scalar("critic_loss", self.critic_loss)

        tf.summary.scalar("Penalty", self.gradient_penalty)
        tf.summary.scalar("learning_rate", self.learning_rate)

        tf.summary.histogram("Distance_fake", self.critic_fake.dist)
        tf.summary.histogram("Distance_real", self.critic_real.dist)

        # visualize  trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        # visualize gradients
        for grad, var in self.gen_grads_and_vars + self.critic_grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradients", grad)

    def training(self, exp_folder, clusters_ratios, checkpoint=None,
                 progress_freq=1000, summary_freq=200, save_freq=5000,
                 validation_freq=500, critic_iter=5, valid_cells_no=500):
        """
        Method that trains the cscGAN.

        Parameters
        ----------
        exp_folder : str
            Path where TF will write the logs, save the model, the t-SNE plots etc.
        clusters_ratios : dict
            Dictionary containing the different cluster names and their
            ratio in the data.
        checkpoint : str, None
            Path to the checkpoint to start from, or None to start training
             from scratch.
            Default is None.
        progress_freq : int
            Period (in steps) between displays of the losses values on the
            standard output.
            Default is 1000.
        summary_freq : int
            Period (in steps) between logs for the Tensorboard.
            Default is 200.
        save_freq : int
            Period (in steps) between saves of the model.
            Default is 5000.
        validation_freq : int
            Period (in steps) between validation measures are computed
            (e.g. t-SNE plots).
            Default is 500.
        critic_iter : int
            Number of training iterations of the critic (inner loop) for each
            iteration on the generator (outer loop).
            Default is 5.
        valid_cells_no :
            Number of cells in the validation set.
            Default is 500.

        Returns
        -------

        """
        exp_name = exp_folder.split('/')[-1]

        # Transform the cluster ratios dictionary into an ordered list
        clusters_ratios = [value for (key, value) in sorted(clusters_ratios.items())]
        clusters_ratios.sort(reverse=True)
        clusters_ratios = np.reshape(clusters_ratios, (1, len(clusters_ratios)))

        # Number of different models to keep (each time a model is saved,
        #  it will overwrite the oldest)
        saver = tf.train.Saver(max_to_keep=1)

        train_supervisor = tf.train.Supervisor(logdir=exp_folder,
                                               save_summaries_secs=0,
                                               saver=None)

        start = time.time()

        # Start the TF session
        with train_supervisor.managed_session() as sess:

            train_feed_dict = {self.generator.is_training: True,
                               self.generator.clusters_ratios: clusters_ratios}

            print("Parameter Count is [ %d ]." % (sess.run(self.parameter_count)))

            # load checkpoint and instantiate the start_step accordingly
            if checkpoint is not None:
                print("Loading model from checkpoint....")
                checkpoint = tf.train.latest_checkpoint(checkpoint)
                saver.restore(sess, checkpoint)
                start_step = sess.run(train_supervisor.global_step)
            else:
                start_step = 0

            critic_fetch = {"train": self.critic_train}

            # Outer loop, one step for the generator and several for the critic
            for step in range(start_step, self.max_steps):

                # small utility function to perform tasks at defined intervals
                def should(freq):
                    return freq > 0 and \
                           ((step + 1) % freq == 0 or
                            step == self.max_steps - 1)

                # Inner loop, for each generator step, several critic steps
                if step > 0:
                    for i_critic in range(critic_iter):
                        sess.run(fetches=critic_fetch, feed_dict=train_feed_dict)

                model_fetches = {"train": self.model_train}

                # Add the corresponding summary tensors to the fetches
                if should(summary_freq):
                    model_fetches["summary"] = train_supervisor.summary_op

                if should(progress_freq):
                    model_fetches["gen_loss"] = self.gen_loss
                    model_fetches["critic_loss"] = self.critic_loss

                results = sess.run(model_fetches, feed_dict=train_feed_dict)

                # Update the summaries for Tensorboard
                if should(summary_freq):
                    print("Recording summary ...")
                    train_supervisor.summary_writer.add_summary(
                        results["summary"], step)

                # Launch the validation steps
                if should(validation_freq):
                    self.validation(sess, valid_cells_no,
                                    exp_folder, step, clusters_ratios)

                # Print out the progresses
                if should(progress_freq):
                    rate = (step + 1) / (time.time() - start)
                    remaining = (self.max_steps - (step + 1)) / rate
                    print("[ %s ] Step number %d ." % (exp_name, step))
                    print("[ %s ] Running rate  %0.3f steps/sec."
                          % (exp_name, rate))
                    print("[ %s ] Estimated remaining time  %d m"
                          % (exp_name, remaining // 60))
                    print("[ %s ] Critic batch loss %0.3f"
                          % (exp_name, results["critic_loss"]))
                    print("[ %s ] Generator batch loss %0.f"
                          % (exp_name, results["gen_loss"]))

                # Save the model
                if should(save_freq):
                    saver.save(sess, os.path.join(exp_folder, "model"),
                               global_step=step)

    def read_valid_cells(self, sess, cells_no):
        """
        Method that reads a given number of cells from the validation set.

        Parameters
        ----------
        sess : Session
            The TF Session in use.
        cells_no : int
            Number of validation cells to read.

        Returns
        -------
        real_cells : numpy array
            Matrix with the required amount of validation cells.
        real_clusters : list
            List containing the corresponding cluster indexes.
        """

        batches_no = int(np.ceil(cells_no // self.batch_size))

        real_cells = []
        real_clusters = []
        for i_batch in range(batches_no):
            test_inputs, test_clusters = sess.run(
                [self.test_cells, self.test_cells_clusters])
            real_cells.append(test_inputs)
            real_clusters.append(test_clusters)

        real_cells = np.array(real_cells, dtype=np.float32)
        real_cells = real_cells.reshape((-1, self.test_cells.shape[1]))

        real_cells = rescale(real_cells, scaling=self.scaling,
                             scale_value=self.scale_value)

        return real_cells, real_clusters

    def generate_cells(self, cells_no, clusters_ratios=None,
                       sess=None, save_path=None, checkpoint=None):
        """
        Method that generate cells from the current model.

        Parameters
        ----------
        cells_no : int or list
            Numbers of cells per cluster to be generated.
            If the clusters_ratios are provided, should be an int (total number of cells).
            If cluster_ratios is None, should be a list of number of cells per cluster.
        clusters_ratios : numpy array
            List containing the different cluster ratios to use for
            the conditional generation.
            Default is None.
        sess : Session
            The TF Session in use.
            If None, a Session is created.
            Default is None.
        save_path : str
            Path in which to write the generated cells.
            If None, the cells are only returned and not written.
            Default is None.
        checkpoint : str /None
            Path to the checkpoint from which to load the model.
            If None, uses the current model loaded in the session.
            Default is None.

        Returns
        -------
        fake_cells : Numpy array
            2-D Array with the gene expression matrix of the generated cells.
        fake_labels : Numpy array
            Array containing the cluster index of the generated cells.
        """

        if sess is None:
            sess = tf.Session()

        if checkpoint is not None:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))

        fake_cells = np.empty((0, self.genes_no), dtype=np.float32)
        fake_labels = np.empty([0, 1], dtype=np.int32)

        if clusters_ratios is None and len(cells_no) > 1:

            for cluster, cells_per_cluster in enumerate(cells_no):
                if int(cells_per_cluster) == 0:
                    continue
                clusters_ratios = np.zeros((1, self.clusters_no), dtype=np.float)
                clusters_ratios[0][cluster] = 1

                fc, fl = self.generate_cells(sess=sess, checkpoint=checkpoint,
                                             cells_no=int(cells_per_cluster),
                                             clusters_ratios=clusters_ratios)

                fake_cells = np.append(fake_cells, fc, axis=0)
                fake_labels = np.append(fake_labels, fl)

        else:

            batches_no = int(np.ceil(cells_no / self.batch_size))

            clusters_ratios_ph = self.generator.clusters_ratios
            fake_labels_tensor = self.generator.input_clusters
            is_training = self.generator.is_training
            fake_cells_tensor = self.generator.fake_outputs

            eval_feed_dict = {is_training: False,
                              clusters_ratios_ph: clusters_ratios}

            for i_batch in range(batches_no):
                fc, fl = sess.run([fake_cells_tensor, fake_labels_tensor],
                                  feed_dict=eval_feed_dict)
                fake_cells = np.append(fake_cells, fc, axis=0)
                fake_labels = np.append(fake_labels, fl)

            fake_labels = fake_labels[0:cells_no]
            fake_cells = fake_cells[0:cells_no]

        rescale(fake_cells, scaling=self.scaling, scale_value=self.scale_value)

        if save_path is not None:
            save_generated_cells(fake_cells, save_path, fake_labels)

        return fake_cells, fake_labels

    def validation(self, sess, cells_no, exp_folder,
                   train_step, clusters_ratios):
        """
        Method that initiates some validation steps of the current model.

        Parameters
        ----------
        sess : Session
            The TF Session in use.
        cells_no : int
            Number of cells to use for the validation step.
        exp_folder : str
            Path to the job folder in which the outputs will be saved.
        train_step : int
            Index of the current training step.
        clusters_ratios : list
            List containing the different cluster ratios to use for the
            conditional generation.

        Returns
        -------

        """

        print("Find tSNE embedding for the generated and the validation cells")
        self.generate_tSNE_image(sess, cells_no, exp_folder,
                                 train_step, clusters_ratios)

    def generate_tSNE_image(self, sess, cells_no, exp_folder,
                            train_step, clusters_ratios):
        """
        Generates and saves a t-SNE plot with real and simulated cells

        Parameters
        ----------
        sess : Session
            The TF Session in use.
        cells_no : int
            Number of cells to use for the real and simulated cells (each)
            used for the plot.
        exp_folder : str
            Path to the job folder in which the outputs will be saved.
        train_step : int
            Index of the current training step.
        clusters_ratios : list
            List containing the different cluster ratios to use for the
            conditional generation.

        Returns
        -------

        """

        tnse_logdir = os.path.join(exp_folder, 'TSNE')
        if not os.path.isdir(tnse_logdir):
            os.makedirs(tnse_logdir)

        # generate fake cells
        fake_cells, fake_clusters = self.generate_cells(
            checkpoint=None,
            cells_no=cells_no,
            clusters_ratios=clusters_ratios,
            sess=sess)

        valid_cells, valid_clusters = self.read_valid_cells(sess, cells_no)

        real_cells_clusters = np.array(valid_clusters, dtype=np.float32).flatten()
        fake_cells_clusters = np.array(fake_clusters, dtype=np.float32).flatten()

        embedded_cells = tsne.fit_transform(
            np.concatenate((valid_cells, fake_cells), axis=0))
        embedded_cells_real = embedded_cells[0:real_cells_clusters.shape[0], :]
        embedded_cells_fake = embedded_cells[real_cells_clusters.shape[0]:, :]

        colormap = cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1, self.clusters_no)]

        plt.clf()
        plt.figure(figsize=(16, 12))

        for i in range(self.clusters_no):
            mask = real_cells_clusters[:] == i

            plt.scatter(embedded_cells_real[mask, 0],
                        embedded_cells_real[mask, 1],
                        c=colors[i], marker='*',
                        label='real_' + str(i))

        for i in range(self.clusters_no):
            mask = fake_cells_clusters[:] == i
            plt.scatter(embedded_cells_fake[mask, 0],
                        embedded_cells_fake[mask, 1],
                        c=colors[i], marker='o',
                        label='fake_' + str(i))

        plt.grid(True)
        plt.legend(loc='lower left',
                   numpoints=1, ncol=3,
                   fontsize=8, bbox_to_anchor=(0, 0))
        plt.savefig(tnse_logdir + '/step_' + str(train_step) + '.jpg')
        plt.close()
