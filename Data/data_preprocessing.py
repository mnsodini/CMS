import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define normalization models

def zscore_preprocess(input_array):
    # Normalizes using zscore scaling ->  x' = (x - μ) / σ
    tensor = tf.convert_to_tensor(input_array, dtype = tf.float32)
    mean = tf.reduce_mean(tensor, axis=1, keepdims=True)
    std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    normalized_tensor = tf.math.divide_no_nan((tensor-mean), std)

    # Masking so unrecorded data = 0
    mask = tf.math.not_equal(tensor, 0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, -1)

    # Outputs normalized pT while preserving original values for eta and phi
    outputs = tf.concat([normalized_tensor[:,:,0,:], tensor[:,:,1,:], tensor[:,:,2,:]], axis=2)
    return tf.reshape(outputs * mask, (-1, 57))


def linear_preprocess(input_array):
    # Normalizing using linear scaling -> x'= (x-xmin) / (xmax - xmin)
    tensor = tf.convert_to_tensor(input_array, dtype = tf.float32)
    min_val = tf.reduce_min(tensor, axis=1, keepdims=True)
    max_val = tf.math.reduce_max(tensor, axis=1, keepdims=True)
    range_val = max_val - min_val
    normalized_tensor = tf.math.divide_no_nan((tensor - min_val), range_val)

    mask = tf.math.not_equal(tensor, 0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, -1)

    outputs = tf.concat([normalized_tensor[:,:,0,:], tensor[:,:,1,:], tensor[:,:,2,:]], axis=2)
    return tf.reshape(outputs * mask, (-1, 57))


def log_preprocess(input_array):
    # Normalizing using log scaling -> x' = log(x+1)
    tensor = tf.convert_to_tensor(input_array, dtype = tf.float32)
    normalized_tensor = tf.math.log(tensor+1) # +1 bc log(0) = nan

    mask = tf.math.not_equal(tensor, 0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, -1)

    outputs = tf.concat([normalized_tensor[:,:,0,:], tensor[:,:,1,:], tensor[:,:,2,:]], axis=2)
    return tf.reshape(outputs * mask, (-1, 57))
