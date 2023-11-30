import os,sys,argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
PKG = str(Path(__file__).resolve(strict=True).parents[1])
if PKG not in sys.path: sys.path.append(PKG)

from utils.config_utils import PipelineManager
from utils.data_utils import BaseDataHandler, parse_data_handler


def main(config_path):
    ''' Initialize data structures for module '''

    # Get data objects
    Manager = PipelineManager(config_path)
    DataHandler      = parse_data_handler(Manager.data_source)
    InventoryHandler = parse_data_handler(Manager.inventory_source)
    create_inventory(Manager, DataHandler, InventoryHandler)


def create_inventory(Manager:PipelineManager, DataHandler:BaseDataHandler, InventoryHandler:BaseDataHandler):
    ''' Main data inventory function 
    
        INPUTS:
            : Manager = <PipelineManager> Control object for pipeline containing parameters
            : DataHandler = <BaseDataHandler> Data I/O object for input data to be inventoried
            : InventoryHandler = <BaseDataHandler> Data I/O object for inventory CSV
    '''

    # Get relevant parameters
    outpath = Manager.inventory_path
    rootdir = Manager.inventory_data_dir
    val_split   = Manager.config.inventory.validation_split
    force_flag  = Manager.config.inventory.force
    seed        = Manager.config.inventory.seed 
    if seed <= 0: seed=None # Use random seed if non-positive
    partition_col = "partition"

    # Skip if existing
    if InventoryHandler.file_exists(outpath) and not force_flag:
        df = InventoryHandler.load_csv(outpath)
        if partition_col in df.columns:
            print(f"Inventory file already created: {outpath}")
            return

    # Get data according to pipeline task
    pipeline_task   = Manager.config.pipeline.task_name
    pipeline_params = Manager.config.pipeline.task_params

    # TODO: Split implementation into task modules (subclasses of PipelineManager?)
    if pipeline_task == "classification":
        # Build inventory of filenames and classes
        colnames = ["filename","class_name","class_label"]
        stratify_cols = ["class_label"]

        # Check class names/labels
        class_names = pipeline_params["CLASS_NAMES"]
        class_labels = pipeline_params["CLASS_LABELS"]
        if len(class_names) != len(set(class_names)):
            raise Exception("Error: Class names must be unique values")
        if len(class_labels) != len(set(class_labels)):
            raise Exception("Error: Class labels must be unique values")
        if len(class_names) != len(class_labels):
            raise Exception("Error: Class names and labels must be same length")
        label_map = {c:l for c,l in zip(class_names,class_labels)}

        # Get valid info to add to dataframe
        fnames = DataHandler.list_all_files(rootdir,filetype="tif")
        dnames = [os.path.basename(os.path.dirname(f)) for f in fnames]
        data = []
        for f,d in zip(fnames,dnames):
            for c in class_names:
                if c == d:
                    data.append((f,c,label_map[c]))
                    break
        # Create and save dataframe
        df = pd.DataFrame(data=data,columns=colnames)
        InventoryHandler.save_csv(df,outpath)


    elif pipeline_task == "segmentation":
        raise NotImplementedError("Implementation not defined for this pipeline task")
    elif pipeline_task == "regression":
        raise NotImplementedError("Implementatio not defined for this pipeline task")
    else:
        raise Exception(f"Invalid pipeline task: '{pipeline_task}'")


    # Add partition information to DataFrame
    if partition_col not in df.columns:
        if val_split >= 1: raise ValueError(f"Validation split must be decimal value (or negative to disable): Got {val_split}")
        df_train, df_valid = train_test_split(df,test_size=val_split,random_state=seed,shuffle=True,stratify=df[stratify_cols])
        df_train["partition"] = "train"
        df_valid["partition"] = "valid"
        df_out = pd.concat([df_train,df_valid],ignore_index=True)
        InventoryHandler.save_csv(df_out,outpath)
        print(f"Inventory created: {outpath}")


if __name__=="__main__":

    parser = argparse.ArgumentParser("Data Inventory Utility for Deep Learning Pipeline")
    parser.add_argument("--config",default="config.json",help="Configuration JSON file with parameters")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise Exception(f"Config file not found: {args.config}")

    main(args.config)

