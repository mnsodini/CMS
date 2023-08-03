print("Importing from 'test.py'")
import os
import sys
# Limit display of TF messages to errors only 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

import models
import losses 
import graphing_module

class MyHyperModel(kt.HyperModel):
    def build(self, hp): 
        latent_dim = hp.Int("latent_dim", min_value=5, max_value=8, step=1)
        learning_rate = hp.Float("learing_rate", min_value = 0.001, max_value = 0.051, step = 0.01)
        CVAE = models.CVAE(losses.SimCLRLoss, temp=0.07, latent_dim=latent_dim)
        CVAE.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True))
        return CVAE

    def fit(self, hp, model, *args, **kwargs): 
        batch_size = hp.Int("batch_size", min_value=20, max_value=1020, step=100)
        return model.fit(*args, batch_size=batch_size, **kwargs)

    
if __name__ == '__main__':
                               
    features_dataset = np.load('Data/max_pt.npz')
    features_train = features_dataset['x_train']
    features_test = features_dataset['x_test']
    features_valid = features_dataset['x_val']
    labels_train = tf.reshape(features_dataset['labels_train'], (-1, 1))
    labels_test = tf.reshape(features_dataset['labels_test'], (-1, 1))
    labels_valid = tf.reshape(features_dataset['labels_val'], (-1, 1))
                               
    tuner = kt.RandomSearch(
        MyHyperModel(),
        kt.Objective("contrastive_loss", direction="min"),
        max_trials=3,
        overwrite=True,
        directory="Data",
        project_name="tune_hypermodel")
                               
    tuner.search(features_train, labels_train, epochs=10)

                               