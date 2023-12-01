import os,sys
import tensorflow as tf

class BaseModel():
    ''' Base class for Keras models compatible with pipeline '''
    __name__ = "base"
    __params__ = {
        "INPUT_SHAPE":[128,128,3],
        "NUM_CLASSES":2,
        "INITIAL_WEIGHTS":None
    }
    __reserved__ = ["__name__","__params__","model","mro"]

    def __init__(self, **kwargs):
        self.model = None
        self._load_params(**kwargs)
        self.set_architecture()

    def _load_params(self,**kwargs):
        ''' Load parameters from config '''
        if self.__params__ is None: raise NotImplementedError("__params__ dict must be initialized for model subclasses")
        # Normalize to lowercase
        norm_params = {p.lower():v for p,v in self.__params__.items()}
        norm_kwargs = {k.lower():v for k,v in kwargs.items()}
        if len(norm_params) != len(self.__params__): raise Exception("Error: Parameter names must be case-insensitive")
        if len(norm_kwargs) != len(kwargs): raise Exception("Error: Parameter names are case-insensitive")
        # Confirm all parameters are defined in config
        for p in norm_params:
            if p.startswith("__") or (p in self.__reserved__):
                raise Exception(f"Parameter matches reserved keyword: {p}. Please update model parameter specification")
            if p not in norm_kwargs:
                orig_params = {param.lower():param for param in self.__params__}
                raise Exception(f"Missing parameter in model parameters: {orig_params[p]}")
        # Confirm all keywords are valid parameters
        for k in norm_kwargs:
            if k not in norm_params:
                orig_kws = {kw.lower():kw for kw in kwargs}
                raise Exception(f"Invalid model parameter provided: {orig_kws[k]}")
        # Assign parameters to object
        for p in norm_params:
            setattr(self,p,norm_kwargs[p])

    def set_architecture(self):
        ''' Set model architecture as instance attribute (self.model) '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")
    
    def get_architecture(self) -> tf.keras.models.Model:
        ''' Get uncompiled model architecture object '''
        if not self.model: raise Exception("Model architecture not defined")
        return self.model

    def load_weights(self, Handler):
        ''' Load initial weights into provided model '''
        if not self.model: raise Exception("Model architecture not defined")
        weights_path = self.model.initial_weights
        if weights_path:
            # Set instance model weights from secondary existing model
            existing_model = Handler.load_model(compile=False)
            self.model.set_weights(existing_model.get_weights())

