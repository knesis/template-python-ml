import os,argparse

from utils.config_utils import PipelineManager

''' Main pipeline script
Must parse execution flow and run operations if desired
Data Inventory, Training, Validation, Prediction
'''

def main(config_path):
    ''' Main deep learning pipeline script '''

    # Create pipeline control object
    Manager = PipelineManager(config_path)

    # TODO: Parse pipeline execution steps and pass manager into them
    # Inventory
    # Train
    # Predict


if __name__=="__main__":

    parser = argparse.ArgumentParser("Deep Learning Pipeline for Keras")
    parser.add_argument("--config",default="config.json",help="Configuration JSON file for pipeline")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise Exception(f"Config file not found: {args.config}")
    
    main(args.config)