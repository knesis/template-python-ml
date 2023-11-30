import os,sys
from keras import layers, models, applications

from base_model import BaseModel

class Xception(BaseModel):
    ''' Xception pre-trained model '''
    __name__ = "xception"
    __params__ = {
        "INPUT_SHAPE":[299,299,3],
        "NUM_CLASSES":2,
        "INITAL_WEIGHTS":None
    }
    def get_architecture(self) -> models.Model:
        ''' Model architecture for Xception '''

        # Specify custom model input
        x = layers.Input(shape=self.input_shape,name="input")

        # Load pretrained model core
        base_model = applications.Xception(include_top=False, weights="imagenet", input_shape=self.input_shape)
        base_model.trainable = False

        # Add custom top layers
        y1 = base_model(x, training=False)
        y1_pool = layers.GlobalAveragePooling2D(name="avg_pool")(y1)
        out = layers.Dense(self.num_classes,activation="softmax",name="output")(y1_pool)

        model = models.Model(x, out, name=self.__name__)
        return model
        