# -*- coding: utf-8 -*-
"""
Circle to C registration.

@author: Xinzhe Luo
"""

import os
import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
import nibabel as nib
import cv2
import transformer, utils, metrics
import logging
import shutil
from scipy import stats
# import pandas as pd
import argparse

# ADDED!
# tf.disable_v2_behavior()

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def _load_data(filename, dtype=np.float32, **kwargs):
    """
    Load from nifty files.

    :param filename: The file name to extract data.
    :param dtype: The output data type.
    :param num_channels: Number of channels.
    :param normalization: Whether to normalize the data using min-max normalization.
    :param clip_value: whether to clip value by the 99 percentile
    :return: An array.
    """
    num_channels = kwargs.pop('num_channels', 1)
    normalization = kwargs.pop('normalization', None)
    clip_value = kwargs.pop('clip_value', True)
    data_format = kwargs.pop('data_format', 'nifty')
    assert data_format in ['nifty', 'png'], "Data format must be 'nifty' or 'png'!"
    assert normalization in [None, 'min-max', 'z-score']

    if data_format == 'nifty':
        img = nib.load(filename)
        data = np.asarray(img.get_fdata(), dtype=dtype)
    elif data_format == 'png':
        img = cv2.imread(filename)
        data = np.asarray(img[..., 0], dtype=dtype)
        image_size = kwargs.pop('image_size', None)
        if image_size is not None:
            data = data[:image_size[0], :image_size[1]]
        data[data != 255] = 0

    if clip_value:
        data = np.clip(data, -np.inf, np.percentile(data, 99))

    if normalization:
        if normalization == 'min-max':
            # min-max normalization
            data_loc = data - np.min(data)
            data_norm = data_loc / np.max(data_loc)

        elif normalization == 'z-score':
            # z-score normalization
            data_norm = stats.zscore(data, axis=None, ddof=1)

        data_expand = np.expand_dims(data_norm, axis=-1)
    else:
        data_expand = np.expand_dims(data, axis=-1)

    if data_format == 'nifty':
        return np.expand_dims(np.tile(data_expand, np.hstack((np.ones(data.ndim), num_channels))), 0), img.affine, \
               img.header
    elif data_format == 'png':
        return np.expand_dims(np.tile(data_expand, np.hstack((np.ones(data.ndim), num_channels))), 0)


def _load_data_from_array(image_array, dtype=np.float32, **kwargs):
    num_channels = kwargs.pop('num_channels', 1)
    normalization = kwargs.pop('normalization', None)
    clip_value = kwargs.pop('clip_value', True)
    assert normalization in [None, 'min-max', 'z-score']

    # Reshape to 2D if the array is 1D
    if len(image_array.shape) == 1:
        image_size = kwargs.pop('image_size', None)
        if image_size is not None:
            image_array = image_array.reshape(image_size)

    data = np.asarray(image_array, dtype=dtype)
    data[data != 255] = 0

    if clip_value:
        data = np.clip(data, -np.inf, np.percentile(data, 99))

    if normalization:
        if normalization == 'min-max':
            # min-max normalization
            data_loc = data - np.min(data)
            data_norm = data_loc / np.max(data_loc)

        elif normalization == 'z-score':
            # z-score normalization
            data_norm = stats.zscore(data, axis=None, ddof=1)

        data_expand = np.expand_dims(data_norm, axis=-1)
    else:
        data_expand = np.expand_dims(data, axis=-1)

    return np.expand_dims(np.tile(data_expand, np.hstack((np.ones(data.ndim), num_channels))), 0)



def _one_hot_label(data, labels=(0, 1)):
    """

    :param data: of shape [1, *vol_shape, 1]
    :param labels:
    :return:
    """
    data = data.squeeze((0, -1))
    n_class = len(labels)
    label = np.zeros((np.hstack((data.shape, n_class))), dtype=np.float32)

    for k in range(1, n_class):
        label[..., k] = (data == labels[k])

    label[..., 0] = np.logical_not(np.sum(label[..., 1:], axis=-1))

    return np.expand_dims(label, 0)


class Trainer(tf.Module): # was: class Trainer(object)
    """
    A unified class to perform various types of demons registration.

    """

    # def __init__(self, input_size=(96, 96), num_channels=1, num_classes=2,
    #              demons_type='compositive', demons_force='moving', **kwargs):
    def __init__(self, input_size=(96, 96), num_channels=1, num_classes=2,
                 demons_type='compositive', demons_force='moving', regularizer='fluid',
                 normalization=None, max_length=2., exp_steps=7, logger=logging):
        """
        Initialize the trainer.

        :param input_size: The input size.
        :param demons_type: The demons type, 'compositive', 'additive' or 'diffeomorphic'.
        :param demons_force: The demons force, 'moving', 'fixed' or 'symmetric'.
        :param kwargs:
        """
        assert demons_type in ['compositive', 'additive', 'diffeomorphic']
        assert demons_force in ['moving', 'fixed', 'symmetric']
        assert regularizer in ['fluid', 'diffusion']

        self.input_size = input_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.demons_type = demons_type
        self.demons_force = demons_force

        self.regularizer = regularizer
        self.normalization = normalization
        self.max_length = max_length
        self.exp_steps = exp_steps if demons_type == 'diffeomorphic' else 0
        self.logger = logger

        # self.kwargs = kwargs
        
        # self.regularizer = self.kwargs.pop('regularizer', 'fluid')
        # assert self.regularizer in ['fluid', 'diffusion']
        # self.normalization = self.kwargs.pop('normalization', None)
        # self.max_length = self.kwargs.pop('max_length', 2.)
        # self.exp_steps = self.kwargs.pop('exp_steps', 4) if demons_type == 'diffeomorphic' else 0
        # self.logger = kwargs.get("logger", logging)

        # initialize placeholder
        with tf.name_scope('inputs'):
            self.target_image = tf.compat.v1.placeholder(tf.float32, [1, input_size[0], input_size[1],
                                                            num_channels], name='target_image')
            self.target_label = tf.compat.v1.placeholder(tf.float32, [1, input_size[0], input_size[1],
                                                            num_classes], name='target_label')
            self.source_image = tf.compat.v1.placeholder(tf.float32, [1, input_size[0], input_size[1],
                                                            num_channels], name='source_image')
            self.source_label = tf.compat.v1.placeholder(tf.float32, [1, input_size[0], input_size[1],
                                                            num_classes], name='source_label')

        # spatial transformation
        with tf.compat.v1.variable_scope('spatial_transformation'):
            # create the vector fields
            self.vector_fields = tf.compat.v1.get_variable('vector_fields', shape=[1, input_size[0], input_size[1], 2],
                                                 initializer=tf.zeros_initializer(), trainable=True)

            self.warped_source_image = transformer.SpatialTransformer(interp_method='linear',
                                                                      name='warp_source_image')([self.source_image,
                                                                                                 self.vector_fields])
            self.warped_source_label = transformer.SpatialTransformer(interp_method='nearest',
                                                                      name='warp_source_label')([self.source_label,
                                                                                                 self.vector_fields])

        # get train_op
        self._train_op = self._get_optimizer()

        # metrics
        with tf.name_scope('metrics'):
            self.mse = tf.reduce_mean(tf.square(self.target_image - self.warped_source_image), name='mse') / tf.reduce_max(tf.square(tf.cast(255, tf.float32)))
            self.dice = metrics.OverlapMetrics(num_classes).averaged_foreground_dice(y_true=self.target_label,
                                                                                     y_seg=self.warped_source_label)

            ### ADDED
            # self.float_mse = tf.reduce_mean(tf.square(self.target_image / tf.reduce_max(self.target_image) - self.warped_source_image / tf.reduce_max(self.warped_source_image)))

            # self.bending_energy = metrics.local_displacement_energy(self.vector_fields, energy_type='bending')
            # self.jacobian_det = metrics.compute_jacobian_determinant(self.vector_fields)
            # self.num_neg_jacob = tf.math.count_nonzero(tf.less_equal(self.jacobian_det, 0), dtype=tf.float32,
            #                                            name='negative_jacobians_number')
            # self.mean_jacobian = tf.reduce_mean(self.jacobian_det, name='mean_jacobian_det')
            # self.min_jacobian = tf.reduce_min(self.jacobian_det, name='min_jacobian_det')
            # self.max_jacobian = tf.reduce_max(self.jacobian_det, name='max_jacobian_det')

    def _get_optimizer(self):
        """
        Get optimizer for registration optimization.

        :return: The train-operation.
        """
        with tf.name_scope('optimizer'):

            # image normalization
            with tf.name_scope('image_normalization'):
                target_norm_image = utils.normalize_image(self.target_image, self.normalization)
                source_norm_image = utils.normalize_image(self.source_image, self.normalization)

            # compute the update fields
            with tf.name_scope('get_update_fields'):
                warped_source_norm_image = transformer.SpatialTransformer()([source_norm_image, self.vector_fields])

                # compute intensity difference
                diff = tf.subtract(target_norm_image, warped_source_norm_image, name='intensity_diff')

                # compute demons forces
                if self.demons_force == 'fixed':
                    jacobian = - utils.compute_image_gradient(target_norm_image, dim=2)
                elif self.demons_force == 'moving':
                    jacobian = - utils.compute_image_gradient(warped_source_norm_image, dim=2)
                elif self.demons_force == 'symmetric':
                    jacobian = - (utils.compute_image_gradient(target_norm_image, dim=2) + utils.compute_image_gradient(
                        warped_source_norm_image, dim=2)) / 2

                # get correspondence update fields
                update_fields = tf.negative((diff * jacobian) / (tf.norm(jacobian, axis=-1,
                                                                         keepdims=True) ** 2 + diff ** 2),
                                            name='update_fields')

                # control update fields by the maximum step length
                update_fields /= tf.reduce_max(tf.norm(update_fields, axis=-1))
                update_fields *= self.max_length

            # fluid-like regularization
            if self.regularizer == 'fluid':
                with tf.name_scope('fluid_regularization'):
                    update_fields = utils.separable_gaussian_filter2d(update_fields, utils.gauss_kernel1d(sigma=1.)) # was separable gaussian filter 3d

            # update the correspondence fields
            with tf.name_scope('update_correspondence_fields'):
                if self.demons_type == 'compositive':
                    corres_fields = update_fields + transformer.SpatialTransformer(
                        interp_method='linear', name='warp_vector_fields')([self.vector_fields,
                                                                            update_fields])

                elif self.demons_type == 'additive':
                    corres_fields = self.vector_fields + update_fields

                elif self.demons_type == 'diffeomorphic':
                    exp_update_fields = tf.expand_dims(transformer.integrate_vec(tf.squeeze(update_fields, 0),
                                                                                 method='ss', nb_steps=self.exp_steps),
                                                       axis=0, name='exp_update_fields')
                    corres_fields = exp_update_fields + transformer.SpatialTransformer(
                        interp_method='linear', name='warp_fields')([self.vector_fields, exp_update_fields])

            # diffusion-like regularization
            if self.regularizer == 'diffusion':
                with tf.name_scope('diffusion_regularization'):
                    corres_fields = utils.separable_gaussian_filter3d(corres_fields, utils.gauss_kernel1d(1.))

            corres_fields = tf.reshape(corres_fields, self.vector_fields.shape)
            train_op = tf.compat.v1.assign(self.vector_fields, corres_fields, validate_shape=True, name='train_op')

        return train_op

    def _initialize(self, training_iters, model_path, restore, prediction_path):
        self.training_iters = training_iters
        self.model_path = model_path
        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(prediction_path)
        abs_model_path = os.path.abspath(model_path)

        # remove the previous directory for model storing and validation prediction
        if not restore:
            self.logger.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            self.logger.info("Removing '{:}'".format(abs_model_path))
            shutil.rmtree(abs_model_path, ignore_errors=True)

        # create a new directory for model storing and validation prediction
        if not os.path.exists(abs_prediction_path):
            self.logger.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(abs_model_path):
            self.logger.info("Allocating '{:}'".format(abs_model_path))
            os.makedirs(abs_model_path)

        init = tf.compat.v1.global_variables_initializer()

        return init

    def save(self, sess, model_path, latest_filename=None, **kwargs):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        :param latest_filename: Optional name for the protocol buffer file that will contains the list of most recent
        checkpoints.
        """

        # saver = tf.train.Saver(**kwargs)
        # save_path = saver.save(sess, model_path, latest_filename=latest_filename)
        # self.logger.info("Model saved to file: %s" % save_path)
        # return save_path

        # ADDED
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, model_path)

        self.logger.info("Model saved to file: %s" % save_path)
        return save_path

    def restore(self, sess, model_path, **kwargs):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver(**kwargs)
        saver.restore(sess, model_path)
        self.logger.info("Model restored from file: %s" % model_path)

    def train(self, data, training_iters, display_steps=1, restore=False, model_path='model_trained',
              prediction_path='prediction', **kwargs):
        """
        Train model on the given data.

        :param data: The data dictionary containing 'target_image', 'target_label', 'source_image', 'source_label',
                     'target_header', 'target_affine'.
        :param training_iters: number of training iterations
        :param display_steps: number of training steps till displaying the metrics
        :param restore: whether to restore the model
        :param model_path: where to save/restore the model
        :param prediction_path: where to store predictions
        :return: metrics - the training metrics;
                 save_path - the saved model path
        """

        save_path = os.path.join(model_path, "model.ckpt")
        init = self._initialize(training_iters, model_path, restore, prediction_path)

        with tf.compat.v1.Session(config=config) as sess:

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.logger.info("Restoring previous model from path: %s" % ckpt.model_checkpoint_path)
                    self.restore(sess, ckpt.model_checkpoint_path)

            metrics = {"MSE": {}, "Dice": {}, "Bending energy": {}, "Mean Jacobian determinant": {},
                       "Min Jacobian determinant": {}, "Max Jacobian determinant": {},
                       "# Negative Jacobians": {}}

            self.logger.info("Start optimization with demons configuration: demons force: {demons_force}, "
                             "demons type: {demons_type}, maximum step length: {max_length:.1f} pixels, "
                             "exponential steps: {exp_steps:.1f}".format(demons_force=self.demons_force,
                                                                         demons_type=self.demons_type,
                                                                         max_length=self.max_length,
                                                                         exp_steps=self.exp_steps)
                             )

            for step in range(training_iters):
                _, mse, dice = sess.run((self._train_op, self.mse, self.dice),
                                       feed_dict={self.target_image: data['target_image'],
                                                  self.target_label: data['target_label'],
                                                  self.source_image: data['source_image'],
                                                  self.source_label: data['source_label']
                                                  }
                                                       )

                # record metrics
                metrics['MSE'][step] = mse
                metrics['Dice'][step] = dice

                if np.isnan(mse) or dice == 0:
                    break

                self.final_mse = mse
                self.final_dice = dice

                self.logger.info("[Iteration {:}], [Metrics] MSE= {:.4f}, Dice= {:.4f}".format(step, mse, dice))

                #ADDED check if display steps is equal to 0
                if display_steps != 0:
                    if step % display_steps == 0:
                        self.store_prediction(sess, data, save_prefix='step%s_' % step, **kwargs)
                        # utils.visualise_metrics([metrics],
                        #                         save_path=self.prediction_path,
                        #                         labels=[self.demons_type + '_demons_' + self.demons_force + '_force'])

            self.logger.info("Optimization Finished!")

            # store final predictions
            self.store_prediction(sess, data, save_prefix='final_', **kwargs)

            self.logger.info("Final Predictions Stored!")

            # save model
            self.save(sess, save_path)

        return metrics, save_path

    def store_prediction(self, sess, data, save_prefix, **kwargs):
        warped_image, warped_label, vector_fields = sess.run((self.warped_source_image,
                                                                            self.warped_source_label,
                                                                            self.vector_fields),
                                                                           feed_dict={
                                                                               self.source_image: data['source_image'],
                                                                               self.source_label: data['source_label']
                                                                               }
                                                                           )

        # utils.save_prediction_png(data['target_image'], data['target_label'], warped_label,
        #                           self.prediction_path, save_prefix=save_prefix)

        utils.save_prediction_nii(np.expand_dims(warped_image.squeeze(0), -2), self.prediction_path, data_type='image',
                                  affine=data['target_affine'], header=data['target_header'],
                                  save_prefix=save_prefix)

        utils.save_prediction_nii(np.expand_dims(warped_label.squeeze(0), -2), self.prediction_path, data_type='label',
                                  affine=data['target_affine'], header=data['target_header'],
                                  save_prefix=save_prefix, **kwargs)

        utils.save_prediction_nii(np.expand_dims(vector_fields.squeeze(0), -2), self.prediction_path,
                                  affine=data['target_affine'], data_type='vector_fields',
                                  header=data['target_header'], save_prefix=save_prefix)


if __name__ == '__main__':
    from datetime import datetime

    t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(description='Diffeomorphic demons registration for 2D images')
    parser.add_argument('--demons_type', default='additive', choices=['additive', 'compositive', 'diffeomorphic'],
                        help='the demons type')
    parser.add_argument('--demons_force', default='moving', choices=['moving', 'fixed', 'symmetric'],
                        help='the demons force')
    parser.add_argument('--regularizer', default='fluid', choices=['fluid', 'diffusion'],
                        help='type of regularization for the displacement field')
    parser.add_argument('--normalization', default='z-score', choices=[None, 'min-max', 'z-score'],
                        help='type of image normalization')
    parser.add_argument('--max_length', default=2., type=float, help='the maximum step length')
    parser.add_argument('--exp_steps', default=7, type=int, help='the number of exponential steps')
    parser.add_argument('--training_iters', default=2000, type=int, help='number of training iterations')
    parser.add_argument('--display_steps', default=200, type=int, help='number of iteration till displaying the result')
    parser.add_argument('--config_name', default=t + '_config.txt', type=str,
                        help='the filename to write down configurations')
    parser.add_argument('--save_prefix', default=t + '_', type=str,
                        help='the save path prefix for the trained model and predictions')
    parser.add_argument('--restore', default=False, action='store_true',
                        help='whether to restore the previous model')
    args = parser.parse_args()

    target_path = './data/C.png'
    source_path = './data/circle.png'

    # set working directory
    print("Working directory: %s" % os.getcwd())
    os.chdir('../')
    print("Working directory changed to %s" % os.getcwd())

    # set saving directory
    model_path = args.save_prefix + '_circle2C_model_trained'
    prediction_path = args.save_prefix + '_circle2C_prediction'

    # load data
    target_image = _load_data(target_path, data_format='png', image_size=(592, 592))
    source_image = _load_data(source_path, data_format='png', image_size=(592, 592))

    target_label = _one_hot_label(target_image, labels=(0, 255))
    source_label = _one_hot_label(source_image, labels=(0, 255))

    training_data = {'target_image': target_image, 'target_label': target_label,
                     'source_image': source_image, 'source_label': source_label,
                     'target_affine': np.eye(4), 'target_header': None
                     }


    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # trainer
        trainer = Trainer(input_size=(592, 592), num_channels=1, num_classes=2,
                          demons_type=args.demons_type, demons_force=args.demons_force,
                          normalization=args.normalization, regularizer=args.regularizer,
                          max_length=args.max_length, exp_steps=args.exp_steps)

        metrics, save_path = trainer.train(training_data,  training_iters=args.training_iters,
                                           display_steps=args.display_steps,
                                           model_path=model_path,
                                           prediction_path=prediction_path,
                                           restore=args.restore)

    # write configurations
    f = open(os.path.join(model_path, args.config_name), 'w+')
    f.write(str(args))