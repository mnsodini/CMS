print('da imports')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
from data_preprocessing import zscore_preprocess
from argparse import ArgumentParser

def make_new_dataset(sample_size, filename, divisions): 
    '''
    Given Monte Carlo datasets and background IDs, make smaller dummy dataset for debugging
    If divisions, splits data to include fixed percent from each label in training data. 
    Otherwise, randomly samples from points. OG split: (W 0.592, QCD 0.338, Z 0.067, tt 0.003)
    '''
    # Load the data and labels files using mmap_mode for efficiency
    data = np.load('datasets_-1.npz', mmap_mode='r')
    labels = np.load('background_IDs_-1.npz', mmap_mode='r')
    test_ix = np.random.choice(data['x_test'].shape[0], size=int(sample_size*0.25), replace=False)

    if divisions == []: 
        # No divisions -> Randomly selects samples to include in smaller batch
        train_ix = np.random.choice(data['x_train'].shape[0], size=sample_size, replace=False)
    else: 
        # divisions provided -> smaller batch has divisions[i] percent of sampels from ith label
        train_ix = []
        train_labels = labels['background_ID_train']
        for label_category in range(4): 
            indices = np.where(train_labels == label_category)[0]
            label_sample_size = int(divisions[label_category] * sample_size)
            if len(indices) < label_sample_size: replacement = True 
            else: replacement = False # If samples avaliable < required -> use replacement 
            indices = np.random.choice(indices, size=label_sample_size, replace=replacement) 
            train_ix.extend(indices)

    # Extract sample_size samples from relevant files
    np.random.shuffle(train_ix)     
    x_train = zscore_preprocess(data['x_train'][train_ix])
    x_test = zscore_preprocess(data['x_test'][test_ix])
    id_train = labels['background_ID_train'][train_ix]
    id_test = labels['background_ID_test'][test_ix]

    # Create and save new .npz with extracted features 
    new_dataset = {'x_train': x_train, 'x_test': x_test, 'labels_train': id_train, 'labels_test': id_test}
    print(f"{filename} successfully saved")
    np.savez(filename, **new_dataset)
    
    
def transform_raw_cms(cms_features): 
    '''
    Given raw cms data of form (N, 33, 3) -> (1 MET, 12 jets, 8 muons, 12 electrons)
    Returns transformed data of form (N, 19, 3) -> (1 MET, 4 electrons, 4 muons, 10 jets)
    '''    
    cms_met = cms_features[:, 0, :]
    cms_jet = cms_features[:, 1:11, :]
    cms_muons = cms_features[:, 13:17, :]
    cms_etron = cms_features[:, 21:25, :]
    
    transformed_features = np.concatenate([cms_met, cms_etron, cms_muons, cms_jet], axis=1)
    return transformed_features 
    
    
def report_file_specs(filename, divisons): 
    data = np.load(filename, mmap_mode='r')
    for key in data.keys():
        print(f"Key: '{key}' Shape: '{data[key].shape}'")
        
    if divisions != []: # Also prints frequency of each label
        label_counts = np.bincount(data['labels_train'].astype(int))
        for label, count in enumerate(label_counts): 
            print(f"Label {label}: {count} occurances")

            
if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()
    parser.add_argument('--sample_size', type=int, required=True)
    parser.add_argument('--filename', type=str, required=True)
    divisions = [0.30, 0.30, 0.30, 0.10]
    args = parser.parse_args()
    
    print("Creating file now:")
    make_new_dataset(args.sample_size, args.filename, divisions)
    print("File Specs:")
    report_file_specs(args.filename, divisions)
    
    
    