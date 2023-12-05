# Keras Deep Learning Pipeline

This repository is intended to streamline the training of deep learning models.
- Training data is inventoried and partitioned into training/validation sets for reproducibility.
- All training hyperparameters are collected in a JSON file which is exposed to the user.
- Trained models can be applied to unlabelled data for prediction
- New models and functionality can be integrated into this pipeline by following defined specifications.

## Quickstart (Ubuntu/Python)

1. This pipeline was developed for Python 3.10 on Ubuntu 22.04 (WSL). Operating behavior may vary in other environments.
2. Clone repository files to local machine and navigate to project directory.
3. Ensure that the required system packages are installed and updated:
    - `sudo apt-get install python3-pip python3-venv`
    - `pip3 install --upgrade pip`
4. Create virtual environment in which to install libraries.
    - `python3 -m venv .venv`
    - `source .venv/bin/activate`
5. Install the required Python libraries into the virtual environment.
    - `pip3 install -r requirements.txt`
6. Update config file with desired parameters.
7. Run main script
    - `python3 main.py --config config.json`


## Config Parameters

- `DATA` = Parameter group for data locations
    - `INVENTORY_DATA_SOURCE` = Source location for data to be compiled in inventory (default = `local`)
    - `INVENTORY_DATA_DIR` = Root directory to search for input data
    - `INVENTORY_CSV_SOURCE` = Source location for the inventory CSV file itself
    - `INVENTORY_CSV_DIR` = Parent directory of the inventory CSV
    - `INVENTORY_CSV_NAME` = Filename for the inventory CSV
    - `TRAINING_MODEL_SOURCE` = Source location for trained Keras models
    - `TRAINING_MODEL_DIR` = Parent directory for trained models
    - `TRAINING_MODEL_NAME` = Unique identifier for the training run. Each training run will create a directory with this name within `TRAINING_MODEL_DIR`. This project directory will contain the final model as well as the training log and checkpoint files.
    - `TRAINING_MODEL_FORMAT` = Output format of trained models. Acceptable values are `keras` (preferred), `h5`, or `tf`.
    - `PREDICTION_DATA_SOURCE` = Source location for the data to be processed for prediction
    - `PREDICTION_DATA_DIR` = Root directory to search for unseen input data for prediction
    - `PREDICTION_OUTPUT_DIR` = Parent directory to store the results of model prediction
- `PIPELINE` = Parameter group for the specific deep learning implementation of the pipeline
    - `TASK_NAME` = Description of the machine learning task to perform
        - Acceptable values include `classification`, `segmentation`, or `regression`
    - `TASK_PARAMS` = Task-specific parameters (will vary according to use-case)
        - Classification-specific parameters:
        - `CLASS_NAMES` = List of class names for training. Data inventory module will assume that input data is organized into sub-folders with these class names.
        - `CLASS_LABELS` = Integer labels corresponding to class names. The order of class labels must match the order of class names.
- `INVENTORY` = Parameter group for the data inventory module
    - `VALIDATION_SPLIT` = Decimal proportion of input data to leave out as the validation set (e.g. 0.2).
    - `SEED` = Integer seed for random number generation. Set as positive value to fix randomization results. Set <= 0 to use random seed.
    - `FORCE` = Boolean flag to force overwriting of inventory file if it exists. Otherwise, the inventory function will return and use an existing file.
- `MODEL` = Parameter group for model architecture and fixed training parameters
    - `MODEL_NAME` = String model identifier enumerated in `src/training/components.py`
    - `MODEL_PARAMS` = Dictionary for model-specific parameters needed to configure architecture
        - Common model parameters:
        - `INPUT_SHAPE` = Dimensionality of input data (Rows, Columns, Channels) to network
        - `NUM_CLASSES` = Number of categories in output data
        - `INITIAL_WEIGHTS` = Filepath to an existing model or set of weights that should be loaded into the network as a starting point. 
            - Used for retraining an existing network. Set to blank string to ignore.
            - Not used for pretrained models. Pretrained models should initialize weights during instantiation
    - `LOSS_NAME` = String identifier for loss function enumerated in `src/training/components.py`
    - `LOSS_PARAMS` = Dictionary for loss-specific parameters. These will be passed to the loss function and must match case.
    - `OPTIMIZER_NAME` = String identifier for optimizer enumerated in `src/training/components.py`
    - `OPTIMIZER_PARAMS` = Dictionary for optimizer-specific parameters. These will be passed to the optimizer and must match case.
        - `LEARNING_RATE` = Initial learning rate to use during training
    - `METRICS` = List of JSON objects defining training metrics. Each JSON object is a single metric which must have the following parameters:
        - `METRIC_NAME` = String identifier for the metric, which must be enumerated in `src/training/components.py`
        - `METRIC_PARAMS` = Dictionary for metric-specific parameters. These will be passed to the metric and must match case.
- `TRAINING` = Parameter group for model training module
    - `BATCH_SIZE` = Number of input data examples to collect for each training batch.
    - `NUM_EPOCHS` = Number of iterations of training. Every training example is shown to the network in batches over the course of one epoch.
    - `LOAD_CHECKPOINT` = Boolean flag to continue training using a model checkpoint saved during the training process.
        - Used to resume training in the event of a power outage or other interruption.
    - `CALLBACKS` = List of JSON objects defining custom callbacks to run during training. Each JSON object is a single callback which must have the following parameters:
        - `CALLBACK_NAME` = String identifier for callbacks, which must be enumerated in `src/training/components.py`
        - `CALLBACK_PARAMS` = Dictionary for callback-specific parameters, if any. These will be passed to the callback function and must match case.
        - Keras callbacks for `ModelCheckpoint` and `CSVLogger` are built-ins for the pipeline and do not need to be explicitly defined
    - `SEED` = Same as above
    - `FORCE` = Boolean flag to force training if a model already exists. Any existing model file will be overwritten.


## Pipeline Development Guide

- Config Parameters
    - New config parameters for `config.json` must also be added to the appropriate parameter object in `utils/config_utils.py`
    - Any new parameter groups must be encapsulated by a subclass of `ParameterObject` and be added to `ConfigTree` in `utils/config_utils.py`
    - References to renamed parameters must be updated in the source code.
- Data Locations
    - Functionality for new data locations must be subclasses of `BaseDataHandler` defined in `utils/data_utils.py`
    - The helper function `parse_data_handler()` must also be updated to reflect the unique string identifier for that data handler
- Model Architectures
    - New model architectures must be subclasses of `BaseModel` defined in `src/models/base_model.py`
    - Any supplemental model parameters must be reflected in the config JSON file
    - The new model must be properly enumerated in `src/training/components.py`
- Training Components
    - Training components include custom loss functions, optimizers, training metrics, and callbacks.
    - Custom components must follow Keras guidelines for implementation 
    - The new components must be properly enumerated in `src/training/components.py`
    - The parameters for any components must be reflected in the config JSON file
- Deep Learning Applications
    - Each deep learning use-case must be encapsulated by a `Task()` object in `utils/pipeline_utils.py`
        - The task object must parse the task name and task parameters defined in the config JSON file
        - The task object must contain methods to perform specialized functionality required by the specific application (e.g. inventory dataframe management, output prediction file parsing, etc.)
    - Each deep learning use-case will also require a unique data generator for loading training data into batches
        - Data generators must be subclasses of `BaseDataGenerator` defined in `src/data/generators.py`
        - The task object must reference the associated data generator class


## File Overview

- `prod` = Production modules
    - `inventory.py` = Data inventory utility to parse available files for training
    - `predict.py` = Model inference utility to evaluate trained model on unseen data
    - `train.py` = Model training utility to tune pretrained or novel architectures
- `src` = Pipeline source code
    - `data` = Internal library for managing datasets
        - `generators.py` = Data generator template for model training
    - `models` = Internal library for model architectures
        - `base_model.py` = Template class for Keras models compatible with pipeline
        - `xception.py` = Implementation of Xception pretrained model for pipeline
    - `training` = Internal library for training configuration
        - `components.py` = Enumeration of training hyperparameters (loss functions, metrics, optimizers, callbacks)
- `utils` = General-purpose utilities for interfacing with external sources
    - `config_utils.py` = Interface to translate config parameters to Python object attributes
    - `data_utils.py` = Interface to read, load, and save data to external locations
    - `pipeline_utils.py` = Specialized control interfaces for different deep learning tasks
- `.gitignore` = Exclude data and non-essential files from repository
- `CHANGELOG.txt` = Internal changelog for pipeline
- `config.json` = Parameters file for pipeline
- `main.py` = Main script for pipeline, processing the main functions from `inventory.py`, `train.py`, and `predict.py` in order
- `README.md` = Pipeline summary documentation
- `requirements.txt` = Required Python libraries to run pipeline