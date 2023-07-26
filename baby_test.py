import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("importing")

# Python library imports
import numpy as np
import h5py
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)

# Other files imports 
import models
import losses 
import graphing_module

def monte_carlo_main(): 
    epochs = 1 
    batch_size = 100
    learning_rate = 0.05
    temp = 0.7
    
    create_latent = False
    create_classifier = False
    make_plots = True
    plot_name = '0725'
    
    print("pulling the data:")
    small_data = np.load('Data/large_divisions.npz')
    features_train = small_data['x_train']
    features_test = small_data['x_test']
    labels_train = tf.reshape(small_data['labels_train'], (-1, 1))
    labels_test = tf.reshape(small_data['labels_test'], (-1, 1))
    
    # Creates model 
    if create_latent: 
        print("making & training model:")
        print("creating the latent space:")
        encoder = models.build_encoder()
        callbacks = [EarlyStopping(monitor='contrastive_loss', patience=10, verbose=1)]
        CVAE = models.CVAE(encoder, losses.SimCLRLoss, temp)
        CVAE.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True))
        CVAE.fit(features_train, labels_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        encoder.save_weights("encoder_weights.h5")
    
    if create_classifier:
        print("creating the classifier:")
        encoder = models.build_encoder()
        encoder.load_weights("encoder_weights.h5")
        Classifier = models.build_classification_head(4)
        Classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True),
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        Classifier.fit(encoder.predict(features_train), labels_train, epochs=epochs, batch_size=batch_size) 

    if make_plots: 
        print("making clustering plots:")
        file_specs = f'F_E{epochs}_B{batch_size}_L{learning_rate}_T{temp}'
        encoder = models.build_encoder()
        encoder.load_weights("encoder_weights.h5")
        test_representation = encoder.predict(features_test)
        # specs = [epochs, batch_size, learning_rate]
        # graphing_module.plot_2D_pca(test_representation, f'{plot_name}_2D_{file_specs}.png', labels = labels_test)
        # graphing_module.plot_3D_pca(test_representation, f'{plot_name}_3D_{file_specs}.png', labels = labels_test)
        # graphing_module.plot_pca_proj(test_representation, f'{plot_name}_PCAProj_{file_specs}.png', labels = labels_test)
        graphing_module.plot_corner_plots(test_representation, f'{plot_name}_CornerPlot.png', labels_test)
        

if __name__ == '__main__':
    monte_carlo_main()
    

