import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("importing")
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

def main(): 
    epochs = 30 
    batch_size = 100
    learning_rate = 0.025
    temp = 0.7
    make_plots = True
    plot_name = 'divided_baby'
    model_name = None
    
    print("pulling the data:")
    small_data = np.load('Data/less_tt.npz')
    features_train = small_data['x_train']
    features_test = small_data['x_test']
    labels_train = small_data['labels_train']
    labels_train = tf.reshape(small_data['labels_train'], (-1, 1))
    labels_test = tf.reshape(small_data['labels_test'], (-1, 1))
    
    # Creates model 
    print("making & training model:")
    encoder = models.build_encoder()
    callbacks = [EarlyStopping(monitor='contrastive_loss', patience=10, verbose=1)]
    CVAE = models.CVAE(encoder, losses.SimCLRLoss, temp)
    CVAE.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True))
    CVAE.fit(features_train, labels_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    if model_name is not None: CVAE.save(model_name)
    
    if make_plots == True: 
        print("making clustering plots:")
        file_specs = f'F_E{epochs}_B{batch_size}_L{learning_rate}_T{temp}'
        test_representation = CVAE.encoder.predict(features_test)
        specs = [epochs, batch_size, learning_rate]
        graphing_module.plot_2D_pca(test_representation, labels_test, f'{plot_name}_2D_{file_specs}.png', specs = specs)
        graphing_module.plot_3D_pca(test_representation, labels_test, f'{plot_name}_3D_{file_specs}.png', specs = specs)
        graphing_module.plot_pca_proj(test_representation, labels_test, f'{plot_name}_PCAProj_{file_specs}.png', specs = specs)
        

if __name__ == '__main__':
    main()
    


