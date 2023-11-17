''' Base class for Keras models used in pipeline
Superset of Keras model
Contains function to return standard Keras compiled model architecture
Contains unique attributes for configuring model parameters
Contains functionality to load existing weights (separate from transfer learning)
Subclasses can implement pre-trained or novel architectures
'''


class BaseModel():
    ''' Base class for Keras models compatible with pipeline '''
    __name__ = "base"
    __params__ = None

    def __init__(self, **kwargs):
        return
