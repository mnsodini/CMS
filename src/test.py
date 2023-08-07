print("Importing from 'test.py'")
import os
import sys
# Limit display of TF messages to errors only 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import data_preprocessing
from tensorflow import keras
import models
import losses 
import graphing_module
from argparse import ArgumentParser
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)

def test_main(full_data, subset_data_name, latent_dim, epochs, batch_size, learning_rate, loss_temp, 
              encoder_name, train, plot, anomaly, anomaly_graph_subset, normalization_type): 
    '''Infastructure for training and plotting CVAE (background specific and with anomalies)'''
    print("=========================")
    print("PULLING DATA FOR TRAINING")
    if full_data: # If full data, pulls from full Delphes training 
        features_dataset = np.load('../data/datasets_-1.npz')
        
        # Pulls and normalizes features data from original file
        if normalization_type == 'max_pt': 
            data_preprocessing.save_normalization_weights(features_dataset['x_train'], 'datasets_-1.npz')
            features_train = data_preprocessing.maxPT_preprocess(features_dataset['x_train'], 'datasets_-1.npz')
            features_test = data_preprocessing.maxPT_preprocess(features_dataset['x_test'], 'datasets_-1.npz')
            features_valid = data_preprocessing.maxPT_preprocess(features_dataset['x_val'], 'datasets_-1.npz')
            
        elif normalization_type == 'zscore':
            features_train = data_preprocessing.zscore_preprocess(features_dataset['x_train'])
            features_test = data_preprocessing.zscore_preprocess(features_dataset['x_test'])
            features_valid = data_preprocessing.zscore_preprocess(features_dataset['x_val'])
        
        features_train = tf.reshape(features_dataset['x_train'], (-1, 57))
        features_test = tf.reshape(features_dataset['x_test'], (-1, 57))   
        features_valid = tf.reshape(features_dataset['x_val'], (-1, 57)) 
        
        # Pulls and reshapes labels dataset from original file 
        labels_dataset = np.load('../data/background_IDs_-1.npz')
        labels_train = tf.reshape(labels_dataset['background_ID_train'], (-1, 1))
        labels_test = tf.reshape(labels_dataset['background_ID_test'], (-1, 1))
        labels_test = tf.reshape(labels_dataset['background_ID_val'], (-1, 1))
        fileanme = 'datasets_-1.npz'
        
    else: # Otherwise use biased batch dataset [0.3 W, 0.3 QCD, 0.2 Z, 0.2 tt]
        features_dataset = np.load('../data/' + subset_data_name)
        features_train = features_dataset['x_train']
        features_test = features_dataset['x_test']
        features_valid = features_dataset['x_val']
        labels_train = tf.reshape(features_dataset['labels_train'], (-1, 1))
        labels_test = tf.reshape(features_dataset['labels_test'], (-1, 1))
        labels_valid = tf.reshape(features_dataset['labels_val'], (-1, 1))
        filename = subset_data_name
        
        folder = f"{epochs}_BatchSize_{batch_size}_LearningRate_{learning_rate}_Temp_{loss_temp}_LatentDim_{latent_dim}"
        folder = "Final_pT"
    
    if train: 
        # Creates CVAE and trains on training data. Saves encoder 
        print("=============================")
        print("MAKING AND TRAINING THE MODEL")
        callbacks = [EarlyStopping(monitor='contrastive_loss', patience=10, verbose=1)]
        CVAE = models.CVAE(losses.SimCLRLoss, temp=loss_temp, latent_dim=latent_dim)
        CVAE.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True))
        history = CVAE.fit(features_train, labels_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                           validation_data=(features_valid, labels_valid))
        graphing_module.plot_contrastive_loss(history, folder, f'0_Loss_Plot.png')
        subfolder = os.path.join(os.path.dirname(__file__), '..', 'model_weights')
        os.makedirs(subfolder, exist_ok=True)
        saved_weights_path = os.path.join(subfolder, encoder_name)
        CVAE.encoder.save_weights(saved_weights_path)
        print(f"MODEL SAVED AT {saved_weights_path}")
    
    encoder = models.build_encoder(latent_dim=latent_dim)
    encoder.load_weights("../model_weights/" + encoder_name)
    
    if plot: 
        print("==============================")
        print("MAKING RELEVANT TRAINING PLOTS")    
        test_representation = encoder.predict(features_test)
        #graphing_module.plot_2D_pca(test_representation, folder, f'1_2D_PCA.png', labels = labels_test)
        #graphing_module.plot_3D_pca(test_representation, folder, f'1_3D_PCA.png', labels = labels_test)
        graphing_module.plot_tSNE(test_representation, folder, f'1_tSNE.png', labels = labels_test)
        #graphing_module.plot_corner_plots(test_representation, folder, f'1_Latent_Corner_Plots.png', labels_test, plot_pca=False)
        #graphing_module.plot_corner_plots(test_representation, folder, f'1_PCA_Corner_Plots.png', labels_test, plot_pca=True)

    if anomaly: 
        print("=============================")
        print("MAKING RELEVANT ANOMALY PLOTS")
        anomaly_dataset = np.load('../data/bsm_datasets_-1.npz')
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
#             anomaly_representation = data_preprocessing.maxPT_preprocess(anomaly_dataset[key], subset_data_name)
            anomaly_representation = data_preprocessing.zscore_preprocess(anomaly_dataset[key])
            anomaly_representation = encoder.predict(anomaly_representation)
            anomaly_labels = tf.fill((anomaly_representation.shape[0], 1), 4.0)
            anomaly_labels = tf.cast(anomaly_labels, dtype=tf.float32)

            # Concatinate background and anomaly to feed into plots
            mixed_representation = np.concatenate([anomaly_representation, background_representation], axis=0)
            mixed_labels = tf.concat([anomaly_labels, background_labels], axis=0)

#             graphing_module.plot_2D_pca(mixed_representation, folder, f'2_{key}_2D_PCA.png', labels=mixed_labels, anomaly=key)
#             graphing_module.plot_3D_pca(mixed_representation, folder, f'2_{key}_3D_PCA.png', labels=mixed_labels, anomaly=key) 
            graphing_module.plot_tSNE(mixed_representation, folder, f'2_{key}_tSNE.png', labels=mixed_labels, anomaly=key)
            graphing_module.plot_corner_plots(mixed_representation, folder, f'2_{key}_Corner_Plot.png', mixed_labels, True, key)
      
    
if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()
    
    # Whether to use full data or a smaller subset sample 
    parser.add_argument('--full_data', type=bool, default=False)
    
    # If using full data, must specify type of normalization intend to use 
    parser.add_argument('--normalization_type', type=str, default='zscore')
    
    # If not using full data, name of smaller dataset to pull from 
    parser.add_argument('--subset_data_name', type=str, default='zscore.npz') 
    
    parser.add_argument('--latent_dim', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.011)
    parser.add_argument('--loss_temp', type=float, default=0.07)
    parser.add_argument('--encoder_name', type=str, default='zscore.h5')
    
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--anomaly', type=bool, default=False)
    
    # If making anomaly graphs, int representing the maximum number of smaples to include from the background 
    parser.add_argument('--anomaly_graph_subset', type=bool, default=False)
    
    args = parser.parse_args()
    test_main(args.full_data, args.subset_data_name, args.latent_dim, args.epochs, args.batch_size, args.learning_rate, 
              args.loss_temp, args.encoder_name, args.train, args.plot, args.anomaly, args.anomaly_graph_subset, 
              args.normalization_type)

        