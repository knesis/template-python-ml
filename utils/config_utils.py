import os,json

''' Base classes for parsing and exposing parameters defined in the config file to the pipeline code 
Actual definition of parameter groups and parameter names 
Possible pipeline manager object with high-level control and lower-level config objects
'''

class ParameterObject():
    ''' General purpose object which stores parameters as attributes '''
    __params__ = None
    __reserved__ = ["__params__","__reserved__","mro"]
    def __init__(self,**kwargs):
        if self.__params__ is None: raise NotImplementedError("__params__ dict must be initialized for subclasses")
        # Normalize to lowercase
        norm_params = {p.lower():v for p,v in self.__params__.items()}
        norm_kwargs = {k.lower():v for k,v in kwargs.items()}
        if len(norm_params) != len(self.__params__): raise Exception("Error: Parameter names must be case-insensitive")
        if len(norm_kwargs) != len(kwargs): raise Exception("Error: Parameter names are case-insensitive")
        # Confirm all parameters are defined in config
        for p in norm_params:
            if p.startswith("__") or (p in self.__reserved__):
                raise Exception(f"Parameter matches reserved keyword: {p}. Please update parameter specification")
            if p not in norm_kwargs:
                orig_params = {param.lower():param for param in self.__params__}
                raise Exception(f"Missing parameter in config: {orig_params[p]}")
        # Confirm all keywords are valid parameters
        for k in norm_kwargs:
            if k not in norm_params:
                orig_kws = {kw.lower():kw for kw in kwargs}
                raise Exception(f"Invalid parameter provided: {orig_kws[k]}")
        # Assign parameters to object
        for p in norm_params:
            setattr(self,p,norm_kwargs[p])


class DataParams(ParameterObject):
    ''' Parameter specification for pipeline data locations '''
    __params__ = {
        "INVENTORY_DATA_DIR":None,
        "INVENTORY_CSV":"inventory.csv",
        "TRAINING_MODEL_DIR":None,
        "TRAINING_MODEL_NAME":None,
        "TRAINING_MODEL_FORMAT":None,
        "PREDICTION_DATA_DIR":None,
        "PREDICTION_OUTPUT_DIR":None
    }

class PipelineParams(ParameterObject):
    ''' Parameter specification for defining machine learning task '''
    __params__ = {
        "TASK_NAME":None,
        "TASK_PARAMS":{}
    }

class InventoryParams(ParameterObject):
    ''' Parameter specification for training data inventory'''
    __params__ = {
        "DATA_SOURCE":"local",
        "VALIDATION_SPLIT":0.2
    }

class ModelParams(ParameterObject):
    ''' Parameter specification for training model architecture'''
    __params__ = {
        "MODEL_NAME":None,
        "MODEL_PARAMS":{},
        "LOSS_NAME":None,
        "LOSS_PARAMS":{},
        "OPTIMIZER_NAME":None,
        "OPTIMIZER_PARAMS":{},
        "METRICS":[]
    }

class TrainingParams(ParameterObject):
    ''' Parameter specification for optimization hyperparameters'''
    __params__ = {
        "LEARNING_RATE":1e-4,
        "BATCH_SIZE":256,
        "NUM_EPOCHS":100,
        "LOAD_CHECKPOINT":False,
        "CALLBACKS":[]
    }


class ConfigTree(ParameterObject):
    ''' Parameter hierarchy of config file '''
    __params__={
        "DATA":DataParams,
        "PIPELINE":PipelineParams,
        "INVENTORY":InventoryParams,
        "MODEL":ModelParams,
        "TRAINING":TrainingParams
    }
    def __init__(self,**kwargs):
        ''' Create child parameter objects for groups in config data'''
        # Assign parameter groups as attributes, with dict values
        super().__init__(**kwargs)
        # Reassign attributes as parameter objects
        norm_grps = {g.lower():c for g,c in self.__params__.items()}
        for g in norm_grps:
            group_data = getattr(self,g)
            ParamCls = norm_grps[g]
            group_params = ParamCls(group_data)
            setattr(self,g,group_params)


class PipelineManager():
    ''' Controller object to manage general training info and paths '''

    def __init__(self,config_path):
        ''' Configure pipeline settings from config '''
        self.filepath = config_path
        self.data = self.load_config(config_path)
        self.config = ConfigTree(**self.data)

        # Specify input/output paths here to reference in modules
        DConfig = self.config.data
        self.inventory_path = os.path.join(DConfig.inventory_data_dir, DConfig.inventory_csv)
        # Model files and save locations
        model_name = DConfig.training_model_name
        model_ext = DConfig.training_model_format.lower() 
        self.model_dir = os.path.join(DConfig.training_model_dir, model_name)
        if model_ext not in ["keras","h5","tf"]:
            raise ValueError(f"Unknown value for training model file format: '{model_ext}'")
        if model_ext == "tf": model_outpath = os.path.join(self.model_dir, model_name)
        else:                 model_outpath = os.path.join(self.model_dir,f"{model_name}.{model_ext}")
        self.model_path = model_outpath
        self.checkpoint_path = os.path.join(self.model_dir,f"{model_name}_chk.{model_ext}")
        self.log_path = os.path.join(self.model_dir,f"{model_name}_log.csv")
        # Prediction data locations
        self.predictions_dir    = DConfig.prediction_data_dir
        self.predictions_path = os.path.join(self.prediction_output_dir,"predictions.csv")
            

    def load_config(self,fpath):
        ''' Load config into object '''
        if not os.path.exists(fpath): 
            raise FileNotFoundError(f"Config path does not exist: {fpath}")
        try:
            with open(fpath) as fobj:
                data = json.load(fobj)
        except: raise Exception(f"Config file not valid: {fpath}")
        return data