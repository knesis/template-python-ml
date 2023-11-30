import os,sys
import tensorflow as tf

# TODO: Refactor for proper abstraction of keras.Model and superset methods
# TODO: Correct to use proper object-oriented programming for get_architecture/load_weights

class BaseModel():
    ''' Base class for Keras models compatible with pipeline '''
    __name__ = "base"
    __params__ = {
        "INPUT_SHAPE":[128,128,3],
        "NUM_CLASSES":2,
        "INITIAL_WEIGHTS":None
    }
    __reserved__ = ["__name__","__params__","mro"]

    def __init__(self, **kwargs):
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


    def get_architecture(self) -> tf.keras.models.Model:
        ''' Instantiate uncompiled model architecture here '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")
    
    def load_weights(self, model:tf.keras.models.Model, ModelHandler):
        ''' Load initial weights into provided model '''
        weights_path = model.initial_weights
        if weights_path:
            existing_model = ModelHandler.load_model(compile=False)
            model.set_weights(existing_model.get_weights())

        return model

