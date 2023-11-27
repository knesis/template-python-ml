import os,argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from utils.config_utils import PipelineManager
from utils.data_utils import parse_data_handler
from prod.inventory import create_inventory

''' Main pipeline script
Must parse execution flow and run operations if desired
Data Inventory, Training, Validation, Prediction
'''

def main(config_path):
    ''' Main deep learning pipeline script '''

    # Create pipeline control object
    Manager = PipelineManager(config_path)

    # Inventory of Training Data
    InventoryHandler = parse_data_handler(Manager.config.data.inventory_data_source)
    create_inventory(Manager,InventoryHandler)

    # TODO: Parse pipeline execution steps and pass manager into them
    # Train
    # Predict


if __name__=="__main__":

    parser = argparse.ArgumentParser("Deep Learning Pipeline for Keras")
    parser.add_argument("--config",default="config.json",help="Configuration JSON file for pipeline")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise Exception(f"Config file not found: {args.config}")
    
    main(args.config)