import os,sys,json
import pandas as pd

from pathlib import Path
PKG = str(Path(__file__).resolve(strict=True).parents[1])
if PKG not in sys.path: sys.path.append(PKG)


from utils.config_utils import ConfigTree
from src.data.generators import *


class PipelineManager():
    ''' Controller object to manage general training info and paths '''

    def __init__(self,config_path):
        ''' Configure pipeline settings from config '''
        self.filepath = config_path
        self.data = self.load_config(config_path)
        self.config = ConfigTree(**self.data)
        self.set_pipeline_paths()
        self.set_task_manager()

    def set_pipeline_paths(self):
        ''' Define input/outpath paths for pipeline data '''

        # Specify data sources for handlers
        DataConfig = self.config.data
        self.inventory_source  = DataConfig.inventory_csv_source
        self.data_source       = DataConfig.inventory_data_source
        self.model_source      = DataConfig.training_model_source
        self.prediction_source = DataConfig.prediction_data_source

        # Inventory filepaths
        self.inventory_data_dir = DataConfig.inventory_data_dir
        self.inventory_path = os.path.join(DataConfig.inventory_csv_dir, 
                                           DataConfig.inventory_csv_name)

        # Training filepaths
        model_root = DataConfig.training_model_dir
        model_name = DataConfig.training_model_name
        model_ext  = DataConfig.training_model_format.lower() 
        self.model_dir = os.path.join(model_root, model_name)

        # Parse trained model file extension
        if model_ext not in ["keras","h5","tf"]:
            raise ValueError(f"Unknown value for training model file format: '{model_ext}'")
        elif model_ext == "tf": 
            model_outpath = os.path.join(self.model_dir, model_name,"")
            model_chkpath = os.path.join(self.model_dir,f"{model_name}_chk","")
        else:                 
            model_outpath = os.path.join(self.model_dir,f"{model_name}.{model_ext}")
            model_chkpath = os.path.join(self.model_dir,f"{model_name}_chk.{model_ext}")

        self.model_path      = model_outpath
        self.checkpoint_path = model_chkpath
        self.log_path        = os.path.join(self.model_dir,f"{model_name}_log.csv")

        # Prediction filepaths
        self.prediction_data_dir  = DataConfig.prediction_data_dir
        self.prediction_path = os.path.join(DataConfig.prediction_output_dir,"predictions.csv")
    
    def set_task_manager(self):
        ''' Assign appropriate controller for the defined learning task '''

        task_name = self.config.pipeline.task_name
        if task_name == "classification":
            self.task = ClassificationTask(self)
        elif task_name == "segmentation":
            raise NotImplementedError("Implementation not defined for this pipeline task")
        elif task_name == "regression":
            raise NotImplementedError("Implementatio not defined for this pipeline task")
        else:
            raise Exception(f"Invalid pipeline task: '{task_name}'")


    def load_config(self,fpath):
        ''' Load config into object '''
        if not os.path.exists(fpath): 
            raise FileNotFoundError(f"Config path does not exist: {fpath}")
        try:
            with open(fpath) as fobj:
                data = json.load(fobj)
        except: raise Exception(f"Config file not valid: {fpath}")
        return data   



class ClassificationTask():
    ''' Pipeline manager with specific functions for classification tasks '''

    def __init__(self, Manager):

        # Set classification-specific information for pipeline
        self.name   = Manager.config.pipeline.task_name
        self.params = Manager.config.pipeline.task_params
        self.class_names  = self.params["CLASS_NAMES"]
        self.class_labels = self.params["CLASS_LABELS"]
        # Inventory dataframe columns
        self.inventory_data_dir = Manager.inventory_data_dir
        self.inventory_cols = ["filename","class_name","class_label"]
        self.stratify_cols  = ["class_label"]
        self.prediction_col = "prediction"
        # Prediction output CSV
        self.prediction_path = Manager.prediction_path
        # Data generator class
        self.generator = ClassificationDataGenerator


    def get_inventory(self, Handler):
        ''' Parse labelled input data (organized by class) for classification 
        
            INPUTS:
                : Handler = <BaseDataHandler> Data I/O object for parsing files
            OUTPUTS:
                : df = <pd.DataFrame> Inventory dataframe
        
        '''
        # Check class names/labels
        rootdir = self.inventory_data_dir
        class_names = self.class_names
        class_labels = self.class_labels
        if len(class_names) != len(set(class_names)):
            raise Exception("Error: Class names must be unique values")
        if len(class_labels) != len(set(class_labels)):
            raise Exception("Error: Class labels must be unique values")
        if len(class_names) != len(class_labels):
            raise Exception("Error: Class names and labels must be same length")
        label_map = {c:l for c,l in zip(class_names,class_labels)}

        # Get valid info to add to dataframe
        fnames = Handler.list_all_files(rootdir,filetype="tif")
        dnames = [os.path.basename(os.path.dirname(f)) for f in fnames]
        data = []
        for f,d in zip(fnames,dnames):
            for c in class_names:
                if c == d:
                    data.append((f,c,label_map[c]))
                    break
        # Create and save dataframe
        df = pd.DataFrame(data=data, columns=self.inventory_cols)
        return df
    
    def process_results(self, results, df, Handler):
        ''' Save output dataframe with predicted labels 
        
        INPUTS:
            : results = <np.ndarray> Output probabilities of model inference (#samples x #classes)
            : df = <pd.DataFrame> Inventory dataframe in which to store predicted labels
            : Handler = <BaseDataHandler> Data I/O object to save output CSV
        '''

        outpath = self.prediction_path
        # Get actual class labels from predicted label indices
        class_labels = self.class_labels
        pred_idx = np.argmax(results,axis=1)
        pred_labels = [class_labels[i] for i in pred_idx]
        # Save dataframe
        df[self.prediction_col] = pred_labels
        Handler.save_csv(outpath)