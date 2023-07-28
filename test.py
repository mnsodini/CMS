print("importing from test.py")
import os
import sys
# Limit display of TF messages to errors only 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import models
import losses 
import graphing_module
import numpy as np
import tensorflow as tf
from tensorflow import keras
from argparse import ArgumentParser
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)
from data_preprocessing import zscore_preprocess

def test_main(full_data, latent_dim, epochs, batch_size, learning_rate, loss_temp, encoder_name, 
               train, plot, anomaly, anomaly_graph_subset): 
    '''Infastructure for training and plotting CVAE (background specific and with anomalies)'''
    
    print("=========================")
    print("PULLING DATA FOR TRAINING")
    if full_data: # If full data, pulls from full Delphes training 
        features_dataset = np.load('Data/datasets_-1.npz')
        features_train = zscore_preprocess(features_dataset['x_train'])
        features_test = zscore_preprocess(features_dataset['x_test'])
        labels_dataset = np.load('Data/background_IDs_-1.npz')
        labels_train = tf.reshape(labels_dataset['background_ID_train'], (-1, 1))
        labels_test = tf.reshape(labels_dataset['background_ID_test'], (-1, 1))
        
    else: # Otherwise use biased batch dataset [0.3 W, 0.3 QCD, 0.2 Z, 0.2 tt]
        features_dataset = np.load('Data/large_divisions.npz')
        features_train = features_dataset['x_train']
        features_test = features_dataset['x_test']
        labels_train = tf.reshape(features_dataset['labels_train'], (-1, 1))
        labels_test = tf.reshape(features_dataset['labels_test'], (-1, 1))
    
    if train: 
        # Creates CVAE and trains on training data
        print("=============================")
        print("MAKING AND TRAINING THE MODEL")
        callbacks = [EarlyStopping(monitor='contrastive_loss', patience=10, verbose=1)]
        CVAE = models.CVAE(losses.SimCLRLoss, temp=loss_temp, latent_dim=latent_dim)
        CVAE.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True))
        CVAE.fit(features_train, labels_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        CVAE.encoder.save_weights(encoder_name)
        print(f"MODEL SAVED AT {encoder_name}")
    
    encoder = models.build_encoder(latent_dim=latent_dim)
    print("Saved_Weights/" + encoder_name)
    encoder.load_weights("Saved_Weights/" + encoder_name)
    folder = f"Epochs_{epochs}_BatchSize_{batch_size}_LearningRate_{learning_rate}_Temp_{loss_temp}_LatentDim_{latent_dim}"
    
    if plot: 
        print("==============================")
        print("MAKING RELEVANT TRAINING PLOTS")    
        test_representation = encoder.predict(features_test)
#         graphing_module.plot_2D_pca(test_representation, folder, f'1_2D_PCA.png', labels = labels_test)
#         graphing_module.plot_3D_pca(test_representation, folder, f'1_3D_PCA.png', labels = labels_test)
        graphing_module.plot_corner_plots(test_representation, folder, f'1_Latent_Corner_Plots.png', labels_test, plot_pca=False)
        graphing_module.plot_corner_plots(test_representation, folder, f'1_PCA_Corner_Plots.png', labels_test, plot_pca=True)

    if anomaly: 
        print("=============================")
        print("MAKING RELEVANT ANOMALY PLOTS")
        anomaly_dataset = np.load('Data/bsm_datasets_-1.npz')
        background_representation = encoder.predict(features_test)
        background_labels = tf.cast(labels_test, dtype=tf.float32)
        
        if anomaly_graph_subset: 
            print("sampling subset of background")
            # Pulls random subset of background to include in anomaly plots
            shuffled_background_indices = tf.random.shuffle(background_indices)[:500000]
            background_representation = tf.gather(background_representation, shuffled_background_indices)
            background_labels = tf.gather(background_labels, shuffled_background_indices)

        for key in anomaly_dataset.keys(): 
            print(f"making plots for {key} anomaly")
            # Compute anomaly representations and define anomaly labels
            anomaly_representation = zscore_preprocess(anomaly_dataset[key])
            anomaly_representation = encoder.predict(anomaly_representation)
            anomaly_labels = tf.fill((anomaly_representation.shape[0], 1), 4.0)
            anomaly_labels = tf.cast(anomaly_labels, dtype=tf.float32)

            # Concatinate background and anomaly to feed into plots
            mixed_representation = np.concatenate([anomaly_representation, background_representation], axis=0)
            mixed_labels = tf.concat([anomaly_labels, background_labels], axis=0)

            graphing_module.plot_2D_pca(mixed_representation, folder, f'2_{key}_2D_PCA.png', labels=mixed_labels, anomaly=key)
            graphing_module.plot_3D_pca(mixed_representation, folder, f'2_{key}_3D_PCA.png', labels=mixed_labels, anomaly=key) 
            graphing_module.plot_corner_plots(mixed_representation, folder, f'2_{key}_Corner_Plot.png', mixed_labels, True, key)
      
    
if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()
    parser.add_argument('--full_data', type=bool, default=False)
    parser.add_argument('--latent_dim', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1082)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--loss_temp', type=float, default=0.07)
    parser.add_argument('--encoder_name', type=str, default='encoder_weights_latent6.h5')
    
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--anomaly', type=bool, default=True)
    parser.add_argument('--anomaly_graph_subset', type=bool, default=False)
    args = parser.parse_args()
    
    test_main(args.full_data, args.latent_dim, args.epochs, args.batch_size, args.learning_rate, args.loss_temp, 
              args.encoder_name, args.train, args.plot, args.anomaly, args.anomaly_graph_subset)

        