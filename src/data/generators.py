import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.transform import resize

# Add internal library to path
from pathlib import Path
PKG = str(Path(__file__).resolve(strict=True).parents[2])
sys.path.append(PKG)

# Import internal modules
from utils.data_utils import BaseDataHandler


class BaseDataGenerator(tf.keras.utils.Sequence):
    ''' Base class for generating batches of training data '''

    def __init__(self, df:pd.DataFrame, handler:BaseDataHandler, input_shape, num_classes, batch_size, training=False, shuffle=False):
        self.df = df
        self.handler = handler
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.training = training
        self.shuffle = shuffle
        # Initialize datapaths and indices
        self.x = self._get_x_info()
        self.y = self._get_y_info()
        self.idxs = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        ''' Return number of batches in dataset '''
        return np.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        ''' Return the next batch of data for training '''
        # Get (shuffled) indices for current batch
        idx_b0 = idx*self.batch_size
        idx_b1 = min((idx+1)*self.batch_size, len(self.x))
        batch_idxs = self.idxs[idx_b0:idx_b1]
        # Get datapaths corresponding to indices
        batch_x = [self.x[i] for i in batch_idxs]
        batch_y = [self.y[i] for i in batch_idxs]
        # Get actual data from batches
        if self.training:
            data_x = self._get_x_data(batch_x)
            data_y = self._get_y_data(batch_y)
            return data_x, data_y
        else:
            data_x = self._get_x_data(batch_x)
            return data_x

    def on_epoch_end(self):
        ''' Shuffle indices every epoch, if desired'''
        if self.shuffle: np.random.shuffle(self.idxs)

    def _get_x_info(self):
        ''' Parse input data paths from dataframe '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def _get_y_info(self):
        ''' Parse labelled output data paths from dataframe '''        
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def _get_x_data(self, batch_x):
        ''' Load batch input data from source '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def _get_y_data(self, batch_y):
        ''' Load batch labelled output data from source '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")
    


class ClassificationDataGenerator(BaseDataGenerator):
    ''' Data generator for image classification batches '''

    def _get_x_info(self):
        ''' Return filenames of input images '''
        xcol = "filename"
        if xcol not in self.df.columns: raise Exception(f"Data column ({xcol}) not in dataframe")
        x = self.df[xcol].to_list()
        return x
    
    def _get_y_info(self):
        ''' Return labels of input images '''
        ycol = "class_labels"
        if ycol not in self.df.columns: raise Exception(f"Label column ({ycol}) not in dataframe")
        y = self.df[ycol].to_list()
        return y
    
    def _get_x_data(self, batch_x):
        ''' Load images into input batch '''
        # Initialize batch dataset
        data_x = np.zeros((self.batch_size,*self.input_shape),dtype="float")
        for i in np.arange(self.batch_size):
            # Load image using data handler method
            fpath = batch_x[i]
            img = self.handler.load_image(fpath)
            # Enforce image dimensionality
            if (len(img.shape)!=3) and (self.input_shape[2]==3):
                img = np.repeat(img[:,:,np.newaxis],self.input_shape[2],axis=2)
            # Resize to input shape
            img = resize(img,self.input_shape,order=2,mode='constant',preserve_range=True)
            # TODO: Augment image here
            data_x[i] = img

        return data_x
    
    def _get_y_data(self, batch_y):
        ''' Format labels for classification '''
        # Convert labels to one-hot representation
        data_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        return data_y