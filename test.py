# Limit display of TF messages to errors only
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Python library imports
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)

# Other files imports 
import models
import losses 
import graphing_module
from Data/data_preprocessing import zscore_preprocess

def fully_supervised(): 
    # Pulls training data from .npz and converts into batches
    print("============")
    print("PULLING DATA")
    features_dataset = np.load('Data/datasets_-1.npz')
    features_train = zscore_preprocess(features_dataset['x_train'])
    features_test = zscore_preprocess(features_dataset['x_test'])
    labels_dataset = np.load('Data/background_IDs_-1.npz')
    labels_train = tf.reshape(labels_dataset['background_ID_train'], (-1, 1))
    labels_test = tf.reshape(labels_dataset['background_ID_test'], (-1, 1))
    
    
    # Creates model 
    print("=============================")
    print("MAKING AND TRAINING THE MODEL")
    encoder = models.build_encoder()
    callbacks = [EarlyStopping(monitor='contrastive_loss', patience=10, verbose=1)]
    CVAE = models.CVAE(encoder, losses.SimCLRLoss, temp=args.loss_temp)
    CVAE.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, amsgrad=True))
    CVAE.fit(features_train, labels_train, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    
    
    print("Make plots argument:", args.make_plots)
    if args.make_plots == True: 
        print("=====================")
        print("MAKING RELEVANT PLOTS")
        file_specs = f'F_E{args.epochs}_B{args.batch_size}_L{args.learning_rate}_T{args.loss_temp}'
        test_representation = CVAE.encoder.predict(features_test)
        graphing_module.plot_2D_pca(test_representation, labels_test, f'{args.plot_name}_2D_{file_specs}.png')
        graphing_module.plot_3D_pca(test_representation, labels_test, f'{args.plot_name}_3D_{file_specs}.png')
        graphing_module.plot_pca_proj(test_representation, labels_test, f'{args.plot_name}_PCAPROJ_{file_specs}.png')
        

    if args.make_anomaly_plots: 
        print("====================")
        print("MAKING ANOMALY PLOTS")
        anomaly_dataset = np.load('Data/bsm_datasets_-1.npz')
        background_representation = CVAE.encoder.predict(features_test)
        background_labels = tf.cast(labels_test, dtype=tf.float32)
        background_indices = tf.range(start=0, limit=background_labels.shape[0], dtype=tf.int32)
        shuffled_background_indices = tf.random.shuffle(background_indices)[:500000]
        
        shuffled_background_representation  = tf.gather(background_representation, shuffled_background_indices)
        shuffled_background_labels = tf.gather(background_labels, shuffled_background_indices)
        
        for key in anomaly_dataset.keys(): 
            print("Making the plot for", key)
            anomaly_representation = zscore_preprocess(anomaly_dataset[key])
            anomaly_representation = CVAE.encoder.predict(anomaly_representation)
            anomaly_labels = tf.fill((anomaly_representation.shape[0], 1), 4.0)
            anomaly_labels = tf.cast(anomaly_labels, dtype=tf.float32)
            mixed_representation = tf.concat([shuffled_background_representation, anomaly_representation], axis=0)
            mixed_labels = tf.concat([shuffled_background_labels, anomaly_labels], axis=0)
            file_specs = f'A_{key}_E{args.epochs}_B{args.batch_size}_L{args.learning_rate}_T{args.loss_temp}'
            graphing_module.plot_3D_pca(mixed_representation, mixed_labels, f'{args.plot_name}_3D_{file_specs}.png', key)


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1082)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--loss_temp', type=float, default=0.07)
    parser.add_argument('--make_plots', type=bool, default=True)
    parser.add_argument('--plot_name', type=str, default='0714')
    parser.add_argument('--make_anomaly_plots', type=bool, default=False)
    args = parser.parse_args()

    fully_supervised() 
