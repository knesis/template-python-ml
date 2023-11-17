import os,json
import pandas as pd

class BaseDataHandler():
    ''' Base class for managing input/output data '''

    def file_exists(self,fpath):
        ''' Helper function to determine if files exist '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def load_csv(self,fpath):
        ''' Load CSV to dataframe '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")
    
    def load_json(self,fpath):
        ''' Load JSON data to dictionary '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def load_image(self,fpath):
        ''' Load TIF image to NumPy array '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")
    
    def load_model(self,fpath):
        ''' Load Keras model from file to object'''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")
    

class LocalDataHandler(BaseDataHandler):
    ''' Data handler for files stored on local system '''

    def file_exists(self, fpath):
        flag = os.path.exists(fpath) and os.path.isfile(fpath)
        return flag
        
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
        
