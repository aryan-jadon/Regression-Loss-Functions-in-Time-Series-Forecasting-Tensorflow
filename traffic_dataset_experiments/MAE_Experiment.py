import datetime as dte
import os
import json
import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model_mae_loss
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model_mae_loss.TemporalFusionTransformer
tf.experimental.output_all_intermediates(True)

with open('traffic_dataset_experiments.json', 'r') as f:
    loss_experiment_tracker = json.load(f)

dataset_name = "traffic"
dataset_folder_path = "traffic_dataset"

name = dataset_name
output_folder = dataset_folder_path

use_tensorflow_with_gpu = True
print("Using output folder {}".format(output_folder))

config = ExperimentConfig(name, output_folder)
formatter = config.make_data_formatter()

expt_name = name
use_gpu = use_tensorflow_with_gpu
model_folder = os.path.join(config.model_folder, "fixed")
data_csv_path = config.data_csv_path
data_formatter = formatter
use_testing_mode = True

num_repeats = 1

if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
    raise ValueError(
        "Data formatters should inherit from" +
        "AbstractDataFormatter! Type={}".format(type(data_formatter)))

# Tensorflow setup
default_keras_session = tf.keras.backend.get_session()

if use_gpu:
    tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)
else:
    tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

print("*** Training from defined parameters for {} ***".format(expt_name))

print("Loading & splitting data...")
raw_data = pd.read_csv(data_csv_path, index_col=0)
train, valid, test = data_formatter.split_data(raw_data)
train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

# Sets up default params
fixed_params = data_formatter.get_experiment_params()
params = data_formatter.get_default_model_params()
params["model_folder"] = model_folder

# Parameter overrides for testing only! Small sizes used to speed up script.
if use_testing_mode:
    fixed_params["num_epochs"] = 15
    params["hidden_layer_size"] = 16
    train_samples, valid_samples = 1000, 100

# Sets up hyper-param manager
print("*** Loading hyperparm manager ***")
opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                   fixed_params, model_folder)

# Training -- one iteration only
print("*** Running calibration ***")
print("Params Selected:")

for k in params:
    print("{}: {}".format(k, params[k]))

best_loss = np.Inf

for _ in range(num_repeats):
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        tf.keras.backend.set_session(sess)
        params = opt_manager.get_next_parameters()
        model = ModelClass(params, use_cudnn=use_gpu)

        if not model.training_data_cached():
            model.cache_batched_data(train, "train", num_samples=train_samples)
            model.cache_batched_data(valid, "valid", num_samples=valid_samples)

        sess.run(tf.global_variables_initializer())
        model.fit()

        val_loss = model.evaluate()

        if val_loss < best_loss:
            opt_manager.update_score(params, val_loss, model)
            best_loss = val_loss

        tf.keras.backend.set_session(default_keras_session)

print("*** Running tests ***")
tf.reset_default_graph()

with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    tf.keras.backend.set_session(sess)
    best_params = opt_manager.get_best_params()
    model = ModelClass(best_params, use_cudnn=use_gpu)

    model.load(opt_manager.hyperparam_folder)

    print("Computing best validation loss")
    val_loss = model.evaluate(valid)

    print("Computing test loss")
    output_map = model.predict(test, return_targets=True)

    targets = data_formatter.format_predictions(output_map["targets"])
    p10_forecast = data_formatter.format_predictions(output_map["p10"])
    p50_forecast = data_formatter.format_predictions(output_map["p50"])
    p90_forecast = data_formatter.format_predictions(output_map["p90"])


    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]


    p10_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p10_forecast),
        0.1)

    p50_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p50_forecast),
        0.5)

    p90_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p90_forecast),
        0.9)

    tf.keras.backend.set_session(default_keras_session)

print("Training completed @ {}".format(dte.datetime.now()))
print("Best validation loss = {}".format(val_loss))
print("Params:")

for k in best_params:
    print(k, " = ", best_params[k])

print("Normalised Quantile Loss for Test Data: P10={}, P50={}, P90={}".format(
    p10_loss.mean(), p50_loss.mean(), p90_loss.mean()))

loss_experiment_tracker.update({
    "Mean Absolute Error p10 Loss": str(p10_loss.mean()),
    "Mean Absolute Error p50 Loss": str(p50_loss.mean()),
    "Mean Absolute Error p90 Loss": str(p90_loss.mean()),
})

with open("traffic_dataset_experiments.json", "w") as outfile:
    json.dump(loss_experiment_tracker, outfile)
