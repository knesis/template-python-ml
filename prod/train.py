import os,sys,argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf

from pathlib import Path
PKG = str(Path(__file__).resolve(strict=True).parents[1])
if PKG not in sys.path: sys.path.append(PKG)

from utils.config_utils import PipelineManager
from utils.data_utils import BaseDataHandler, parse_data_handler
from src.training.components import *
from src.data.generators import *


def main(config_path):
    ''' Initialize data structures for module '''

    # Get data objects
    Manager = PipelineManager(config_path)
    DataHandler         = parse_data_handler(Manager.data_source)
    InventoryHandler    = parse_data_handler(Manager.inventory_source)
    ModelHandler        = parse_data_handler(Manager.model_source)
    train_model(Manager, DataHandler, InventoryHandler, ModelHandler)


def train_model(Manager:PipelineManager, DataHandler:BaseDataHandler, 
                InventoryHandler:BaseDataHandler, ModelHandler:BaseDataHandler):
    ''' Main model training function 
    
        INPUTS:
            : Manager = <PipelineManager> Control object for pipeline containing parameters
            : DataHandler = <BaseDataHandler> Data I/O object for input data to be loaded for training
            : InventoryHandler = <BaseDataHandler> Data I/O object to load inventory CSV
            : ModelHandler = <BaseDataHandler> Data I/O object for loading/saving model
    '''

    # Extract relevant parameters
    invpath = Manager.inventory_path
    outpath = Manager.model_path
    chkpath = Manager.checkpoint_path
    logpath = Manager.log_path

    ModelConfig     = Manager.config.model
    TrainingConfig  = Manager.config.training
    batch_size      = TrainingConfig.batch_size
    num_epochs      = TrainingConfig.num_epochs
    use_checkpoint  = TrainingConfig.load_checkpoint
    callbacks       = TrainingConfig.callbacks
    force_flag      = TrainingConfig.force
    seed            = TrainingConfig.seed
    if seed > 0: tf.keras.utils.set_random_seed(seed)

    # Skip model training if output completed
    if ModelHandler.file_exists(outpath) and not force_flag:
        print(f"Trained model already exists: {outpath}")
        return

    # Setup model according to architecture and parameters
    model = setup_model(Manager, ModelHandler)

    # Load data inventory
    df = InventoryHandler.load_csv(invpath)
    df_train = df[df["partition"]=="train"]
    df_valid = df[df["partition"]=="valid"]

    # Parse pipeline task for proper data generator class
    pipeline_task = Manager.config.pipeline.pipeline_task
    if pipeline_task == "classification":
        Generator = ClassificationDataGenerator
    elif pipeline_task == "segmentation":
        raise NotImplementedError("Implementation not defined for this pipeline task")
    elif pipeline_task == "regression":
        raise NotImplementedError("Implementation not defined for this pipeline task")
    else:
        raise Exception(f"Invalid pipeline task: '{pipeline_task}'")

    # Create data generators
    TrainingData    = Generator(df_train, DataHandler, ModelConfig, batch_size, training=True, shuffle=True)
    ValidationData  = Generator(df_valid, DataHandler, ModelConfig, batch_size, training=True, shuffle=True)

    # Load callbacks and training settings
    custom_callbacks = []
    for cbk in callbacks:
        callback_name = cbk["CALLBACK_NAME"]
        callback_params = cbk["CALLBACK_PARAMS"]
        if callback_name not in ALL_CALLBACKS:
            raise Exception(f"Error: Invalid callback: {callback_name}")
        callback_obj = ALL_CALLBACKS[callback_name](**callback_params)
        custom_callbacks.append(callback_obj)

    # TODO: Callbacks save files locally and should be modified to use handler functionality
    callback_chk = tf.keras.callbacks.ModelCheckpoint(chkpath, monitor="val_loss", verbose=1, save_best_only=True)   
    callback_log = tf.keras.callbacks.CSVLogger(logpath,append=use_checkpoint)
    training_callbacks = [callback_chk, callback_log, *custom_callbacks]
    
    # Train data
    ModelHandler.makedirs(os.path.dirname(outpath))
    model.fit(x=TrainingData, validation_data=ValidationData, batch_size=batch_size, epochs=num_epochs, 
              callbacks=training_callbacks, verbose=1)

    # Save output model and inventory
    ModelHandler.save_model(model, outpath)
    outpath_inventory = os.path.join(Manager.model_dir, f"{os.path.basename(Manager.model_dir)}_inventory.csv")
    ModelHandler.save_csv(df, outpath_inventory)

    # TODO: Validation statistics 

def setup_model(Manager:PipelineManager, ModelHandler:BaseDataHandler):
    ''' Helper function to configure model architecture and optimization '''

    chkpath = Manager.checkpoint_path
    use_checkpoint = Manager.config.training.load_checkpoint

    ModelConfig         = Manager.config.model
    model_name          = ModelConfig.model_name
    model_params        = ModelConfig.model_params
    loss_name           = ModelConfig.loss_name
    loss_params         = ModelConfig.loss_params   
    optimizer_name      = ModelConfig.optimizer_name
    optimizer_params    = ModelConfig.optimizer_params
    metric_info         = ModelConfig.metrics

    # Get model architecture (or checkpoint)
    if use_checkpoint:
        model = ModelHandler.load_model(chkpath,compile=False)
    else:
        if model_name not in ALL_MODELS:
            raise Exception(f"Error: Model not implemented: {model_name}")
        ModelCls = ALL_MODELS[model_name](**model_params)
        ModelCls.load_weights(ModelHandler)
        model = ModelCls.get_architecture()

    # Configure loss function from config
    if loss_name not in ALL_LOSSES:
        raise Exception(f"Error: Invalid loss function: {loss_name}")
    loss_fn = ALL_LOSSES[loss_name](**loss_params)

    # Configure optimizer from config
    if optimizer_name not in ALL_OPTIMIZERS:
        raise Exception(f"Error: Invalid optimizer: {optimizer_name}")
    optimizer = ALL_OPTIMIZERS[optimizer_name](**optimizer_params)

    # Configure metrics from config
    metrics = []
    for m in metric_info:
        metric_name = m["METRIC_NAME"]
        metric_params = m["METRIC_PARAMS"]
        if metric_name not in ALL_METRICS:
            raise Exception(f"Error: Invalid metric: {metric_name}")
        metric_obj = ALL_METRICS[metric_name](**metric_params)
        metrics.append(metric_obj)

    # Compile model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model


if __name__=="__main__":

    parser = argparse.ArgumentParser("Model Training Utility for Deep Learning Pipeline")
    parser.add_argument("--config",default="config.json",help="Configuration JSON file with parameters")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise Exception(f"Config file not found: {args.config}")

    main(args.config)

