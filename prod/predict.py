import os,sys,argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from pathlib import Path
PKG = str(Path(__file__).resolve(strict=True).parents[1])
if PKG not in sys.path: sys.path.append(PKG)

from utils.pipeline_utils import PipelineManager
from utils.data_utils import BaseDataHandler, parse_data_handler
from src.data.generators import *


def main(config_path):
    ''' Initialize data structures for module '''

    # Get data objects
    Manager = PipelineManager(config_path)
    ModelHandler        = parse_data_handler(Manager.model_source)
    PredictionHandler   = parse_data_handler(Manager.prediction_source)
    predict_data(Manager, PredictionHandler, ModelHandler)


def predict_data(Manager:PipelineManager, PredictionHandler:BaseDataHandler, ModelHandler:BaseDataHandler):
    ''' Main prediction function 
    
        INPUTS:
            : Manager = <PipelineManager> Control object for pipeline containing parameters
            : PredictionHandler = <BaseDataHandler> Data I/O object for unseen data for model inference
            : ModelHandler = <BaseDataHandler> Data I/O object for loading/saving model
    '''

    # Extract relevant parameters
    modelpath = Manager.model_path
    outpath   = Manager.prediction_path
    rootdir   = Manager.prediction_data_dir
    ModelConfig = Manager.config.model

    # Skip model training if output completed
    if PredictionHandler.file_exists(outpath):
        print(f"Prediction CSV already exists: {outpath}")
        return

    # Load existing trained model
    model = ModelHandler.load_model(modelpath)

    # Create simple data inventory for prediction data
    fnames = PredictionHandler.list_all_files(rootdir,filetype="tif")
    df = pd.DataFrame(data=[fnames],columns=["filename"])

    # Create data generator
    batch_size = 32
    Generator = Manager.task.generator
    PredictionData = Generator(df, PredictionHandler, ModelConfig, batch_size, training=False, shuffle=False)

    # Run model inference and get predictions
    results = model.predict(x=PredictionData)
    Manager.task.process_results(results,df,PredictionHandler)


if __name__=="__main__":

    parser = argparse.ArgumentParser("Model Inference Utility for Deep Learning Pipeline")
    parser.add_argument("--config",default="config.json",help="Configuration JSON file with parameters")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise Exception(f"Config file not found: {args.config}")

    main(args.config)

