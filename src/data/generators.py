import numpy as np
import pandas as pd
import tensorflow as tf


''' Data generator objects for loading batches of data for training
Agnostic to actual data location (handled by separate data IO object)
Includes augmentation (may be separate module)
Subclasses of base generator can be tailored to specific use cases, such as classification, regression, segmentation, etc.
'''

class BaseDataGenerator(tf.keras.utils.Sequence):
    ''' Base class for generating batches of training data '''

    def __init__(self, df, handler, input_shape, num_classes, batch_size, training=False, shuffle=False):
        self.df = df
        self.handler = handler
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.training = training
        self.shuffle = shuffle
        # Initialize datapaths and indices
        self.x = self._get_x_paths()
        self.y = self._get_y_paths()
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

    def _get_x_paths(self):
        ''' Parse input data paths from dataframe '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def _get_y_paths(self):
        ''' Parse labelled output data paths from dataframe '''        
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def _get_x_data(self, batch_x):
        ''' Load batch input data from source '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")

    def _get_y_data(self, batch_y):
        ''' Load batch labelled output data from source '''
        raise NotImplementedError("Error: Abstract method from base class. This must be implemented in subclasses")
