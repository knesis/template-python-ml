import os,sys,argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from pathlib import Path
PKG = str(Path(__file__).resolve(strict=True).parents[1])
if PKG not in sys.path: sys.path.append(PKG)

from utils.config_utils import PipelineManager
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
        print(f"Trained model already exists: {outpath}")
        return

    # Load existing trained model
    model = ModelHandler.load_model(modelpath)

    # Create simple data inventory for prediction data
    fnames = PredictionHandler.list_all_files(rootdir,filetype="tif")
    df = pd.DataFrame(data=[fnames],columns=["filename"])


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

    # Create data generator
    batch_size = 32
    PredictionData = Generator(df, PredictionHandler, ModelConfig, batch_size, training=False, shuffle=False)

    # Run model inference and get predictions
    results = model.predict(x=PredictionData)

    # Parse results according to pipeline task
    if pipeline_task == "classification":

        # Get predicted labels and save
        class_labels = Manager.config.pipeline.pipeline_params["CLASS_LABELS"]
        pred_idx = np.argmax(results,axis=1)
        pred_labels = [class_labels[i] for i in pred_idx]
        df["prediction"] = pred_labels
        PredictionHandler.save_csv(outpath)
        
    elif pipeline_task == "segmentation":
        raise NotImplementedError("Implementation not defined for this pipeline task")
    elif pipeline_task == "regression":
        raise NotImplementedError("Implementation not defined for this pipeline task")
    else:
        raise Exception(f"Invalid pipeline task: '{pipeline_task}'")


if __name__=="__main__":

    parser = argparse.ArgumentParser("Model Inference Utility for Deep Learning Pipeline")
    parser.add_argument("--config",default="config.json",help="Configuration JSON file with parameters")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise Exception(f"Config file not found: {args.config}")

    main(args.config)

