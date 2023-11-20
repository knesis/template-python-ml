import tensorflow as tf

''' Collection of training pipeline components which can be referenced in config 
Includes metrics, losses, optimizers, and callbacks
With creation of several custom components, may need to split into losses.py, optimizers.py, metrics.py, callbacks.py
'''

ALL_LOSSES = {
    "binary_crossentropy":tf.keras.losses.BinaryCrossentropy,
    "categorical_crossentropy":tf.keras.losses.CategoricalCrossentropy,
    "mse":tf.keras.losses.MeanSquaredError
}
ALL_OPTIMIZERS = {
    "adam":tf.keras.optimizers.Adam,
    "sgd":tf.keras.optimizers.SGD,
    "rmsprop":tf.keras.optimizers.RMSprop
}
ALL_METRICS = {
    "accuracy":tf.keras.metrics.Accuracy,
    "auc":tf.keras.metrics.AUC,
    "mean_iou":tf.keras.metrics.MeanIoU
}
ALL_CALLBACKS = {
    "early_stopping":tf.keras.callbacks.EarlyStopping,
    "reduce_lr_plateau":tf.keras.callbacks.ReduceLROnPlateau
}