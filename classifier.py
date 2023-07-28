print("importing from classifier.py") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import h5py
import models
import numpy as np
import tensorflow as tf
from tensorflow import keras
from argparse import ArgumentParser
from graphing_module import plot_ROC
from data_preprocessing import zscore_preprocess


def classifier_main(latent_dim, encoder_name, folder):
    '''
    Builds and trains DNN classifier to distinguish background (0 labeled) vs anomalies (1 labeled)
    Plots ROC to visualize efficiacy 
    '''
    # Builds the encoder using pretrained weights 
    encoder = models.build_encoder()
    encoder.load_weights(encoder_name)
    
    print("=======================")
    print("PULLING BACKGROUND DATA")
    background_features_dataset = np.load('Data/large_divisions.npz')
    background_features_train = encoder.predict(background_features_dataset['x_train'])
    background_features_test = encoder.predict(background_features_dataset['x_test'])
    
    # Classifies all background as 0 (non anomolous)
    background_labels = tf.fill((features_train.shape[0], 1), 0.0)
    background_labels = tf.cast(background_labels, dtype=tf.float32)
    
    print("======================")
    print("PULLING ANOMALOUS DATA")
    anomaly_dataset = np.load('Data/bsm_datasets_-1.npz')
    anomaly_list = ['ato4l', 'leptoquark', 'hChToTauNu', 'hToTauTau']
    
    #Creates dictionary mapping anomalies to latent space representations
    anomaly_mapping = {} 
    for anomaly in anomaly_list: 
        prediction = zscore_preprocess(anomaly_dataset[anomaly])
        prediction = encoder.predict(prediction)
        anomaly_mapping[anomaly] = prediction

    anomaly_test_classes, background_test_classes = {}, {}
    for anomaly in anomaly_mapping: 
        # Finda anomaly latent representation and classifies as 1 (anomolous)
        anomaly_representation = anomaly_mapping[anomaly]
        anomaly_labels = tf.fill((anomaly_representation.shape[0], 1), 1.0)
        anomaly_labels = tf.cast(anomaly_labels, dtype=tf.float32)
        
        # Concatinates background (0 labeled) and anomalous (1 labeled) for classifier training
        mixed_representation = tf.concat([features_train, anomaly_representation], axis=0)
        mixed_labels = tf.concat([background_labels, anomaly_labels], axis=0)
        
        # Trains classifier using specified anomaly
        print("==============================")
        print(f"TRAINING USING {anomaly} DATA")
        classifier = model.build_classification_head(latent_dim)
        classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05, amsgrad=True),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits = False), metrics=['accuracy'])
        classifier.fit(mixed_representation, mixed_labels, epochs=5, batch_size=1026) 

        # Uses classifier to label anomalous and testing background for evaluation. Updates graphing dicts 
        print("=============================")
        print(f"TESTING {anomaly} CLASSIFIER")
        anomaly_test_classes[anomaly] = tf.concat([classifier.predict(anomaly_mapping[a]) for a in anomaly_list], axis=0)
        background_test_classes[anomaly] = classifier.predict(features_test)
        
    # Plots ROC curves and saves file 
    plot_ROC(background_test_classes, anomaly_test_classes, folder, f'3_ROC.png', anomaly)
        

if __name__ == '__main__':
    # Parses terminal command 
    # Includes standard args used for storing graphs within correct subfolder
    parser = ArgumentParser()
    parser.add_argument('--full_data', type=bool, default=False)
    parser.add_argument('--latent_dim', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1082)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--loss_temp', type=float, default=0.07)
    parser.add_argument('--encoder_name', type=str, default='new_encoder.h5')
    args = parser.parse_args()
    
    folder = f"E{epochs}_B{batch_size}_L{learning_rate}_T{loss_temp}_L{latent_dim}"
    classifier_main(args.latent_dim, args.encoder_name, folder)
    
    
    
    