# Limit display of TF messages to errors only
print("Importing for predict.py")

import os
import sys 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Python library imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from argparse import ArgumentParser

# Other files imports 
from models import build_encoder
import graphing_module


def main(cms_filename, encoder_filename, make_plots, plot_name): 
    # Download data and find latent space representations
    print("Downloading the data:")
    cms_data = np.load('Data/processed_cms.npz')
    cms_features = cms_data['dataset'][:100000]

    print("Setting encoder weights and predicting:")
    cms_representations = encoder.predict(cms_features)

    if make_plots: 
        graphing_module.plot_2D_pca(cms_representations, args.plot_name)
        graphing_module.plot_3D_pca(cms_representations, args.plot_name)
        

if __name__ == '__main__':
    #Parses terminal command
    parser = ArgumentParser()
    parser.add_argument('--cms_filename', type=str, default='processed_cms.npz')
    parser.add_argument('--encoder_filename', type=str, default='encoder_weights.h5')
    parser.add_argument('--make_plots', type=bool, default=True)
    parser.add_argument('--plot_name', type=str, default='RawCMS')
    args = parser.parse_args()
    
    main(args.cms_filename, args.encoder_filename, args.make_plots, args.plot_name)

