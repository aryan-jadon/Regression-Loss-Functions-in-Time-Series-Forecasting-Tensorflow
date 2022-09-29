# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Generic helper functions used across codebase."""

import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from tensorflow import keras
from tensorflow.keras import layers


# Generic.
def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.

  Args:
    input_type: Input type of column to extract
    column_definition: Column definition list for experiment
  """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError('Invalid number of columns for {}'.format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.

  Args:
    data_type: DataType of columns to extract.
    column_definition: Column definition to use.
    excluded_input_types: Set of input types to exclude

  Returns:
    List of names for columns with data type specified.
  """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


# Loss functions.
def tensorflow_quantile_loss(y, y_pred, quantile):
    """Computes quantile loss for tensorflow.

  Standard quantile loss as defined in the "Training Procedure" section of
  the main TFT paper

  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)

  Returns:
    Tensor for quantile loss.
  """

    # Checks quantile
    if quantile < 0 or quantile > 1:
        raise ValueError(
            'Illegal quantile value={}! Values should be between 0 and 1.'.format(
                quantile))

    prediction_underflow = y - y_pred
    q_loss = quantile * tf.maximum(prediction_underflow, 0.) + (
            1. - quantile) * tf.maximum(-prediction_underflow, 0.)

    return tf.reduce_sum(input_tensor=q_loss, axis=-1)


def numpy_normalised_quantile_loss(y, y_pred, quantile):
    """Computes normalised quantile loss for numpy arrays.

  Uses the q-Risk metric as defined in the "Training Procedure" section of the
  main TFT paper.

  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)

  Returns:
    Float for normalised quantile loss.
  """
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
                      + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

    quantile_loss = weighted_errors.mean()
    normaliser = y.abs().mean()

    return 2 * quantile_loss / normaliser


# OS related functions.
def create_folder_if_not_exist(directory):
    """Creates folder if it doesn't exist.

  Args:
    directory: Folder path to create.
  """
    # Also creates directories recursively
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Tensorflow related functions.
def get_default_tensorflow_config(tf_device='gpu', gpu_id=0):
    """Creates tensorflow config for graphs to run on CPU or GPU.

  Specifies whether to run graph on gpu or cpu and which GPU ID to use for multi
  GPU machines.

  Args:
    tf_device: 'cpu' or 'gpu'
    gpu_id: GPU ID to use if relevant

  Returns:
    Tensorflow config.
  """

    if tf_device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # for training on cpu
        tf_config = tf.compat.v1.ConfigProto(
            log_device_placement=False, device_count={'GPU': 0})

    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print('Selecting GPU ID={}'.format(gpu_id))

        tf_config = tf.compat.v1.ConfigProto(log_device_placement=False)
        tf_config.gpu_options.allow_growth = True

    return tf_config


def save(tf_session, model_folder, cp_name, scope=None):
    """Saves Tensorflow graph to checkpoint.

  Saves all trainiable variables under a given variable scope to checkpoint.

  Args:
    tf_session: Session containing graph
    model_folder: Folder to save models
    cp_name: Name of Tensorflow checkpoint
    scope: Variable scope containing variables to save
  """
    # Save model
    if scope is None:
        saver = tf.compat.v1.train.Saver()
    else:
        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=100000)

    save_path = saver.save(tf_session,
                           os.path.join(model_folder, '{0}.ckpt'.format(cp_name)))
    print('Model saved to: {0}'.format(save_path))


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
    """Loads Tensorflow graph from checkpoint.

  Args:
    tf_session: Session to load graph into
    model_folder: Folder containing serialised model
    cp_name: Name of Tensorflow checkpoint
    scope: Variable scope to use.
    verbose: Whether to print additional debugging information.
  """
    # Load model proper
    load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

    print('Loading model from {0}'.format(load_path))

    print_weights_in_checkpoint(model_folder, cp_name)

    initial_vars = set(
        [v.name for v in tf.compat.v1.get_default_graph().as_graph_def().node])

    # Saver
    if scope is None:
        saver = tf.compat.v1.train.Saver()
    else:
        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=100000)
    # Load
    saver.restore(tf_session, load_path)
    all_vars = set([v.name for v in tf.compat.v1.get_default_graph().as_graph_def().node])

    if verbose:
        print('Restored {0}'.format(','.join(initial_vars.difference(all_vars))))
        print('Existing {0}'.format(','.join(all_vars.difference(initial_vars))))
        print('All {0}'.format(','.join(all_vars)))

    print('Done.')


def print_weights_in_checkpoint(model_folder, cp_name):
    """Prints all weights in Tensorflow checkpoint.

  Args:
    model_folder: Folder containing checkpoint
    cp_name: Name of checkpoint

  Returns:

  """
    load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

    print_tensors_in_checkpoint_file(
        file_name=load_path,
        tensor_name='',
        all_tensors=True,
        all_tensor_names=True)


from keras import backend


class MeanBiasError(keras.losses.Loss):
    def __init__(self, name="mean_bias_error"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        bias_error = tf.math.subtract(y_true, y_pred)
        size_value = tf.dtypes.cast(tf.size(y_true), tf.float32)
        mbe_loss = backend.mean(backend.sum(bias_error) / size_value)
        return mbe_loss


class RelativeAbsoluteError(keras.losses.Loss):
    def __init__(self, name="relative_absolute_error"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        true_mean = backend.mean(y_true)
        squared_error_num = backend.sum(backend.abs(tf.math.subtract(y_true, y_pred)))
        squared_error_den = backend.sum(backend.abs(tf.math.subtract(y_true, true_mean)))
        rae_loss = tf.math.divide(squared_error_num, squared_error_den)
        return rae_loss


class RelativeSquaredError(keras.losses.Loss):
    def __init__(self, name="relative_squared_error"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        squared_error_num = backend.sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
        squared_error_den = backend.sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
        rse_loss = tf.math.divide(squared_error_num, squared_error_den)
        return rse_loss


class NormalizedRootMeanSquaredError(keras.losses.Loss):
    def __init__(self, name="normalized_root_mean_squared_error"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        squared_error = tf.math.square(tf.math.subtract(y_true, y_pred))
        sum_squared_error = backend.sum(squared_error)
        size_value = tf.dtypes.cast(tf.size(y_true), tf.float32)
        rmse = tf.math.sqrt(sum_squared_error / size_value)
        nrmse_loss = tf.math.divide(rmse, backend.std(y_pred))
        return nrmse_loss


class RelativeRootMeanSquaredError(keras.losses.Loss):
    def __init__(self, name="relative_root_mean_squared_error "):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        num = backend.sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
        den = backend.sum(tf.math.square(y_pred))
        squared_error = tf.math.divide(num, den)
        rrmse_loss = backend.sqrt(squared_error)
        return rrmse_loss


class RootMeanSquaredLogarithmicError(keras.losses.Loss):
    def __init__(self, name="root_mean_squared_logarithmic_error "):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        first_log = tf.math.log(backend.maximum(y_pred, backend.epsilon()) + 1.0)
        second_log = tf.math.log(backend.maximum(y_true, backend.epsilon()) + 1.0)

        return tf.math.sqrt(backend.mean(
            tf.math.squared_difference(first_log, second_log), axis=-1
        ))
