import os,sys
import tensorflow as tf

from pathlib import Path
PKG = str(Path(__file__).resolve(strict=True).parents[2])
if PKG not in sys.path: sys.path.append(PKG)

from src.models import xception


ALL_MODELS = {
    "xception":xception.Xception
}
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