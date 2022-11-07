# Regression Loss Functions Performance Evaluation in Time Series Forecasting using Temporal Fusion Transformers

[![DOI](https://zenodo.org/badge/539248750.svg)](https://zenodo.org/badge/latestdoi/539248750)

```
This repository contains the implementation of paper Temporal Fusion Transformers for Interpretable 
Multi-horizon Time Series Forecasting with different loss functions in Tensorflow. 
We have compared 14 regression loss functions performance on 4 different datasets. 
Summary of experiment with instructions on how to replicate this experiment can be find below.
```

## About Temporal Fusion Transformers
Authors: Bryan Lim, Sercan Arik, Nicolas Loeff and Tomas Pfister

Paper Link: https://arxiv.org/pdf/1912.09363.pdf 

> Abstract - Multi-horizon forecasting problems often contain a complex mix of inputs -- including static (i.e. time-invariant) 
> covariates, known future inputs, and other exogenous time series that are only observed historically -- without any 
> prior information on how they interact with the target. While several deep learning models have been proposed for 
> multi-step prediction, they typically comprise black-box models which do not account for the full range of inputs 
> present in common scenarios. In this paper, we introduce the Temporal Fusion Transformer (TFT) -- a novel 
> attention-based architecture which combines high-performance multi-horizon forecasting with interpretable insights 
> into temporal dynamics. To learn temporal relationships at different scales, the TFT utilizes recurrent layers for 
> local processing and interpretable self-attention layers for learning long-term dependencies. 
> The TFT also uses specialized components for the judicious selection of relevant features and a series of gating layers 
> to suppress unnecessary components, enabling high performance in a wide range of regimes. On a variety of real-world datasets, 
> we demonstrate significant performance improvements over existing benchmarks, and showcase three practical 
> interpretability use-cases of TFT.

Majority of this repository work is taken from - https://github.com/google-research/google-research/tree/master/tft.

## Experiments Summary

### Cite This Repository Work

BibTex
```
@software{jadon_aryan_2022_7126804,
  author       = {Jadon, Aryan},
  title        = {{Regression Loss Functions Performance Evaluation 
                   in Time Series Forecasting using Temporal Fusion
                   Transformers}},
  month        = sep,
  year         = 2022,
  note         = {{https://github.com/aryan-jadon/Regression-Loss-Functions-in-Time-Series-Forecasting-Tensorflow}},
  publisher    = {Zenodo},
  version      = {0.1.1},
  doi          = {10.5281/zenodo.7126804},
  url          = {https://doi.org/10.5281/zenodo.7126804}
}
```

APA
```
Jadon, A. (2022). Regression Loss Functions Performance Evaluation in Time Series Forecasting using Temporal Fusion Transformers (Version 0.1.1) [Computer software]. https://doi.org/10.5281/zenodo.7126804
```

![Summary of Loss Functions](https://github.com/aryan-jadon/Regression-Loss-Functions-in-Time-Series-Forecasting-Tensorflow/blob/main/loss_functions_plots/Loss-Functions-Summary.png)

### How To Replicate This Experiment

#### Downloading Data and Running Default Experiments

The key modules for experiments are organised as:

* **data\_formatters**: Stores the main dataset-specific column definitions, along with functions for data transformation and normalization. For compatibility with the TFT, new experiments should implement a unique ``GenericDataFormatter`` (see **base.py**), with examples for the default experiments shown in the other python files.
* **expt\_settings**: Holds the folder paths and configurations for the default experiments,
* **libs**: Contains the main libraries, including classes to manage hyperparameter optimisation (**hyperparam\_opt.py**), the main TFT network class (**tft\_model.py**), and general helper functions (**utils.py**)

Scripts are all saved in the main folder, with descriptions below:

* **run.sh**: Simple shell script to ensure correct environmental setup.
* **script\_download\_data.py**: Downloads data for the main experiment and processes them into csv files ready for training/evaluation.
* **script\_train\_fixed\_params.py**: Calibrates the TFT using a predefined set of hyperparameters, and evaluates for a given experiment.
* **script\_hyperparameter\_optimisation.py**: Runs full hyperparameter optimization using the default random search ranges defined for the TFT.

Our four default experiments are divided into ``volatility``, ``electricity``, ``traffic``, and``favorita``. 
To run these experiments, first download the data, and then run the relevant training routine.

#### Step 1: Download data for default experiments
To download the experiment data, run the following script:

```bash
python3 -m script_download_data $EXPT $OUTPUT_FOLDER
```

where ``$EXPT`` can be any of {``volatility``, ``electricity``, ``traffic``, ``favorita``}, and ``$OUTPUT_FOLDER`` denotes the root folder in which experiment outputs are saved.

#### Step 2: Train and evaluate network
To train the network with the optimal default parameters, run:

```bash
python3 -m script_train_fixed_params $EXPT $OUTPUT_FOLDER $USE_GPU 
```

where ``$EXPT`` and ``$OUTPUT_FOLDER`` are as above, ``$GPU`` denotes whether to run with GPU support (options are {``'yes'`` or``'no'``}).


For full hyperparameter optimization, run:

```bash
python3 -m script_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes
```

where options are as above.

### Running Experiments with Loss Functions

#### Move the Downloaded Dataset to their Respective Experiment Folder

Run Experiment Script 

```bash
python3 running_experiments.py
```
