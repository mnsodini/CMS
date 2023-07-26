import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("importing")
import sys

# Python library imports
import numpy as np
import h5py
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras import layers
from tensorflow.keras import activations

import models
import losses 
from graphing_module import plot_ROC

# Pulls files from Data subfolder 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(current_dir, "Data")
sys.path.append(data_folder_path)
from data_preprocessing import zscore_preprocess


def main():
    print("creating the classifier:")
    encoder = models.build_encoder()
    encoder.load_weights("encoder_weights.h5")
    
    print("pulling background data:")
    small_data = np.load('Data/large_divisions.npz')
    features_train = encoder.predict(small_data['x_train'])
    features_test = encoder.predict(small_data['x_test'])
    background_labels = tf.fill((features_train.shape[0], 1), 0.0)
    background_labels = tf.cast(background_labels, dtype=tf.float32)
    
    print("pulling anomalous data:")
    anomaly_dataset = np.load('Data/bsm_datasets_-1.npz')
    anomaly_list = ['ato4l', 'leptoquark', 'hChToTauNu', 'hToTauTau']
    
    #Creates dictionary mapping anomalies to latent representations
    anomaly_mapping = {}
    for anomaly in anomaly_list: 
        prediction = zscore_preprocess(anomaly_dataset[anomaly])
        prediction = encoder.predict(prediction)
        anomaly_mapping[anomaly] = prediction

    for anomaly in anomaly_mapping: 
        print(f"++++++++++++++++++++++++")
        print(f"Creating {anomaly} plots")
        
        # Combines background (0) and anomalous (1) data to train classifier 
        anomaly_representation = anomaly_mapping[anomaly]
        anomaly_labels = tf.fill((anomaly_representation.shape[0], 1), 1.0)
        anomaly_labels = tf.cast(anomaly_labels, dtype=tf.float32)
        mixed_representation = tf.concat([features_train, anomaly_representation], axis=0)
        mixed_labels = tf.concat([background_labels, anomaly_labels], axis=0)
        
        # Builds and trains new classifier per anomaly
        print("Training classifier")
        classifier = build_classifier()
        classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05, amsgrad=True),
                       loss=tf.keras.losses.BinaryCrossentropy(from_logits = False), metrics=['accuracy'])
        classifier.fit(mixed_representation, mixed_labels, epochs=5, batch_size=1026) 

        # Uses trained classifier to classify anomalous and testing background for evaluation
        print("Predicting classes for testing")

        anomaly_test_classes = {a: classifier.predict(anomaly_mapping[a]) for a in anomaly_list}
        background_test_classes = classifier.predict(features_test)
        
    
        # Plots ROC and saves
        plot_ROC(anomaly_test_classes, background_test_classes, f'0725_Anomaly_{anomaly}_ROC.png', anomaly)
        
    
def build_classifier(): 
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),  
        tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),  
        tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu),  
        tf.keras.layers.Dense(1, activation='sigmoid')])
    return model
    

if __name__ == '__main__':
    main()