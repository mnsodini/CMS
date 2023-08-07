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
import data_preprocessing

class Train_Encoder(kt.HyperModel):
    def build(self, hp): 
        latent_dim = hp.Int("latent_dim", min_value=5, max_value=8, step=1)
        learning_rate = hp.Float("learing_rate", min_value = 0.001, max_value = 0.051, step = 0.01)
        CVAE = models.CVAE(losses.SimCLRLoss, temp=0.07, latent_dim=latent_dim)
        CVAE.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True))
        return CVAE

    def fit(self, hp, model, *args, **kwargs): 
        batch_size = hp.Int("batch_size", min_value=20, max_value=1020, step=100)
        return model.fit(*args, batch_size=batch_size, **kwargs)

class Train_Classifier(kt.HyperModel): 
    def build(self, hp): 
        learning_rate = hp.Float("learing_rate", min_value = 0.001, max_value = 0.051, step = 0.01)
        classifier = models.build_classification_head(6)
        classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits = False), metrics=['accuracy'])
        return classifier

    def fit(self, hp, classifier, *args, **kwargs): 
        batch_size = hp.Int("batch_size", min_value=32, max_value=1020, step=100)
        return classifier.fit(*args, batch_size=batch_size, **kwargs) 
    
if __name__ == '__main__':
    train_search = False 
    
    if train_search: 
        features_dataset = np.load('../data/zscore.npz')
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
        
    else: 
        encoder = models.build_encoder(6)
        encoder.load_weights("../model_weights/" + 'zscore.h5')
    
        background_features_train = np.load('../data/zscore.npz')['x_train']
        background_features_train = encoder.predict(background_features_train)
        background_labels = tf.fill((background_features_train.shape[0], 1), 0.0)
        background_labels = tf.cast(background_labels, dtype=tf.float32)

        # 'hChToTauNu' best performing anomaly for training - assumes using for hyperparam tuning 
        anomaly_dataset = np.load('../data/bsm_datasets_-1.npz')
        anomaly_representation = data_preprocessing.zscore_preprocess(anomaly_dataset['hChToTauNu'])
        anomaly_representation = encoder.predict(anomaly_representation)
        
        anomaly_labels = tf.fill((anomaly_representation.shape[0], 1), 1.0)
        anomaly_labels = tf.cast(anomaly_labels, dtype=tf.float32)
        
        mixed_representation = tf.concat([background_features_train, anomaly_representation], axis=0)
        mixed_labels = tf.concat([background_labels, anomaly_labels], axis=0)
        
        tuner = kt.RandomSearch(
            Train_Classifier(),
            kt.Objective("accuracy", direction="min"),
            max_trials=3,
            overwrite=True,
            project_name="tune_hypermodel")

        tuner.search(mixed_representation, mixed_labels, epochs=10)
