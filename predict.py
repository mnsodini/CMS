print("Importing for predict.py")
import os
import sys 
import h5py
import numpy as np
import graphing_module
from argparse import ArgumentParser
from models import build_encoder

current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(current_dir, "Data")
sys.path.append(data_folder_path)
from data_preprocessing import zscore_preprocess


def predict_main(encoder_name, sample_size, random_sampling, detector_filter=None): 
    '''
    Predicts classifications on raw CMS data based on pre-trained encoder 
    Args - sample_size (int): number of events to include in graphs 
           random_sampling (bool): if true, randomly samples from cms
           detector_filter (None or str): if given, only plots raw CMS data that passed through specific detected 
    '''
    print("=========================")
    print("PULLING DATA FOR PLOTTING")
    raw_cms_dataset = np.load('Data/cms_delphes_filtered.npz')
    cms_features = raw_cms_dataset['dataset']
    
    if random_sampling: # Randomly selects sample_size num of rows to include in features
        random_ix = np.random.choice(cms_features.shape[0], sample_size, replace=False)
        cms_features = cms_features[random_ix, :]
    else: # Pulls the first sample_size num of rows
        cms_features = cms_features[:sample_size]
    
    encoder = build_encoder()
    encoder.load_weights(encoder_name)
    cms_representations = encoder.predict(cms_features)
    
    if detector_filter is not None: 
        # Only plots raw CMS data that was detected from specific detector 
        raw_cms_file = h5py.File('Data/raw_cms.h5', 'r')
        raw_labels = raw_cms_file[detector_filter]
        raw_labels = np.array(raw_labels)[:sample_size]
        raw_labels = raw_labels.astype(int)
        graphing_module.plot_2D_pca(cms_representations, f"{args.plot_name}_2D_PCA", labels = raw_labels)
        graphing_module.plot_3D_pca(cms_representations, f"{args.plot_name}_3D_PCA", labels = raw_labels)

    else: 
        graphing_module.plot_2D_pca(cms_representations, f"{args.plot_name}_2D_PCA")
        graphing_module.plot_3D_pca(cms_representations, f"{args.plot_name}_3D_PCA")
        

if __name__ == '__main__':
    #Parses terminal command
    parser = ArgumentParser()
    # Standard args used for storing graphs within correct subfolder
    parser.add_argument('--full_data', type=bool, default=False)
    parser.add_argument('--latent_dim', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1082)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--loss_temp', type=float, default=0.07)
    parser.add_argument('--encoder_name', type=str, default='new_encoder.h5')
    
    # Function specific args
    parser.add_argument('--sample_size', type=int, default=500000)
    parser.add_argument('--random_sampling', type=bool, default=True)
    parser.add_argument('--detector_filter', default=None)

    args = parser.parse_args()
    file_specs = f"E{epochs}_B{batch_size}_L{learning_rate}_T{loss_temp}_L{latent_dim}"

    main(args.cms_filename, args.encoder_filename, args.make_plots, args.plot_name)

