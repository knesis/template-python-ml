import os,argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from utils.config_utils import PipelineManager
from utils.data_utils import parse_data_handler
from prod.inventory import create_inventory
from prod.train import train_model
from prod.predict import predict_data


def main(config_path):
    ''' Main deep learning pipeline script '''

    # Create pipeline control object
    Manager = PipelineManager(config_path)

    # Inventory of Training Data
    DataHandler = parse_data_handler(Manager.data_source)
    InventoryHandler = parse_data_handler(Manager.inventory_source)
    create_inventory(Manager, DataHandler, InventoryHandler)

    # Model Training
    ModelHandler = parse_data_handler(Manager.model_source)
    train_model(Manager, DataHandler, InventoryHandler, ModelHandler)

    # Prediction
    PredictionHandler = parse_data_handler(Manager.prediction_source)
    predict_data(Manager, PredictionHandler, ModelHandler)


if __name__=="__main__":

    parser = argparse.ArgumentParser("Deep Learning Pipeline for Keras")
    parser.add_argument("--config",default="config.json",help="Configuration JSON file for pipeline")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise Exception(f"Config file not found: {args.config}")
    
    main(args.config)