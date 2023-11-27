import os,json
import numpy as np
import pandas as pd
import tifffile
import tensorflow as tf

class BaseDataHandler():
    ''' Base class for managing input/output data '''

    # List files
    def file_exists(self,fpath:str):
        ''' Helper function to determine if files exist '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def list_all_files(self, rootdir:str, filter_str:str, filetype:str):
        ''' Get list of all files in location '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    # Read input data
    def load_csv(self,fpath:str):
        ''' Load CSV to dataframe '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")
    
    def load_json(self,fpath:str):
        ''' Load JSON data to dictionary '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def load_image(self,fpath:str):
        ''' Load TIF image to NumPy array '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")
    
    def load_model(self,fpath:str):
        ''' Load Keras model from file to object'''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    # Write output data
    def save_csv(self, df:pd.DataFrame, fpath:str):
        ''' Save dataframe to CSV '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def save_json(self, data:dict, fpath:str):
        ''' Save dict to JSON file '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def save_image(self, img:np.ndarray, fpath:str):
        ''' Save NumPy array to TIF image '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def save_model(self, model:tf.keras.models.Model, fpath:str):
        ''' Save Keras model to output location '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")



class LocalDataHandler(BaseDataHandler):
    ''' Data handler for files stored on local system '''

    def file_exists(self, fpath:str):
        flag = os.path.exists(fpath) and os.path.isfile(fpath)
        return flag
        
    def list_all_files(self, rootdir, filter_str="", filetype=""):
        ''' Get list of all files in directory '''

        # Walk through the directory recursively
        all_files = []
        for dirpath, _, fnames in os.walk(rootdir):                
            valid_files = [os.path.join(dirpath,f) for f in fnames 
                           if filter_str in os.path.join(dirpath,f)
                           if os.path.join(dirpath,f).endswith(filetype)]
            all_files.extend(valid_files)
        return all_files

    # Read input data
    def load_csv(self, fpath):
        ''' Load CSV to dataframe '''

        if not self.file_exists(fpath): raise FileNotFoundError(f"File does not exist: {fpath}")
        try: df = pd.read_csv(fpath)
        except: raise Exception(f"Cannot parse CSV file to dataframe: {fpath}")
        return df

    def load_json(self,fpath):
        ''' Load JSON data to dictionary '''

        if not self.file_exists(fpath): raise FileNotFoundError(f"File does not exist: {fpath}")
        try:
            with open(fpath) as fobj:
                data = json.loads(fobj.read())
        except: raise Exception(f"Could not parse JSON file: {fpath}")
        return data
    
    def load_image(self, fpath):
        ''' Load image TIF to NumPy array '''
        
        if not self.file_exists(fpath): raise FileNotFoundError(f"File does not exist: {fpath}")
        try: img = tifffile.imread(fpath)
        except: raise Exception(f"Could not parse TIF file: {fpath}")
        return img
        
    def load_model(self, fpath):
        ''' Load Keras model from local filesystem '''
        
        if not self.file_exists(fpath): raise FileNotFoundError(f"File does not exist: {fpath}")
        try: model = tf.keras.models.load_model(fpath,compile=True)
        except: raise Exception(f"Could not parse model file: {fpath}")
        return model

    # Write output data
    def save_csv(self, df:pd.DataFrame, fpath:str):
        ''' Save dataframe to CSV '''
        df.to_csv(fpath,index=False)

    def save_json(self, data:dict, fpath:str):
        ''' Save dict to JSON file '''
        with open(fpath,'w') as fobj:
            json.dump(data,fobj,indent=4)

    def save_image(self, img:np.ndarray, fpath:str):
        ''' Save NumPy array to TIF image '''
        tifffile.imwrite(fpath,img,compression="DEFLATE")

    def save_model(self, model:tf.keras.models.Model, fpath:str):
        ''' Save Keras model to output location '''
        tf.keras.models.save_model(model,fpath)
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")





# General utility functions 

def parse_data_handler(handler_name) -> BaseDataHandler:
    ''' Helper function to get handler object from keyword '''

    if handler_name == "local":
        handler = LocalDataHandler()
    else:
        raise Exception(f"Invalid handler name provided: {handler_name}")
    return handler