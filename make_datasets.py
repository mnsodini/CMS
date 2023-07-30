print("Importing from 'make_datasets.py'")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
import tensorflow as tf
import data_preprocessing
from argparse import ArgumentParser

def make_montecarlo_dataset(sample_size, filename, divisions): 
    '''
    Given Monte Carlo datasets and background IDs, make smaller dummy dataset for debugging
    If divisions, splits data to include fixed percent from each label in training data. 
    Otherwise, randomly samples from points. OG split: (W 0.592, QCD 0.338, Z 0.067, tt 0.003)
    '''
    # Load the data and labels files using mmap_mode for efficiency
    data = np.load('Data/datasets_-1.npz', mmap_mode='r')
    labels = np.load('Data/background_IDs_-1.npz', mmap_mode='r')
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

    # Extract sample_size samples from relevant files and saves (μ, σ) and configs.py 
    np.random.shuffle(train_ix)
    data_preprocessing.save_normalization_weights(data['x_train'][train_ix], filename)
    data_preprocessing.save_normalization_weights(data['x_test'][test_ix], filename)
    
    # Normalizes train and test data 
    x_train = data_preprocessing.zscore_preprocess(data['x_train'][train_ix], filename)
    x_test  = data_preprocessing.zscore_preprocess(data['x_test'][test_ix], filename)
    id_train = labels['background_ID_train'][train_ix]
    id_test = labels['background_ID_test'][test_ix]

    # Create and save new .npz with extracted features 
    new_dataset = {'x_train': x_train, 'x_test': x_test, 'labels_train': id_train, 'labels_test': id_test}
    if not os.path.exists('Data'): 
        os.makedirs('Data')
    subfolder_path = os.path.join('Data', filename)
    np.savez(subfolder_path, **new_dataset)
    
    
def make_raw_cms_dataset(cms_filename, new_filename): 
    '''
    Given raw CMS, converts to npz file with appropriate transofmrations: 
    '''
    raw_cms_file = h5py.File(cms_filename, 'r')
    raw_data = raw_cms_file['full_data_cyl']

    delphes_filter = raw_cms_file['L1_SingleMu22']
    dataset_np = np.array(raw_data)
    filter_np  = np.array(delphes_filter)
    dataset_np = dataset_np[filter_np]
    
    # Reordering and reshaping 
    dataset_np = data_preprocessing.transform_raw_cms(dataset_np)
    
    # Convert from computer int embedding to meaningful float rep
    dataset_np = data_preprocessing.convert_to_float(dataset_np)
    
    # Phi Shift so range from [0, 2π] to [-π, π]
    dataset_np = tf.reshape(dataset_np, (-1, 19, 3, 1))
    dataset_np = data_preprocessing.phi_shift(dataset_np)
    
    # Zscore normalization along pT axis 
    dataset_np = tf.reshape(dataset_np, (-1, 19, 3, 1))
    dataset_np = tf.cast(dataset_np, dtype=tf.float32)
    
    data_preprocessing.save_normalization_weights(dataset_np, new_filename)
    dataset_np = data_preprocessing.zscore_preprocess(dataset_np, new_filename)
    
    subfolder_path = os.path.join('Data', filename)
    np.savez(subfolder_path, dataset=dataset_np)
    raw_cms_file.close()
    print("datset converted and saved as NPZ file")
    
    
def report_file_specs(filename, divisions):
    print("loading")
    data = np.load('Data/' + filename, mmap_mode='r')
    print('printing da keys')
    for key in data.keys(): 
        print(f"Key: '{key}' Shape: '{data[key].shape}'")

    if divisions != []: # prints frequency of each label
        label_counts = data['labels_train'].astype(int)
        label_counts = np.bincount(label_counts)
        for label, count in enumerate(label_counts): 
            print(f"Label {label}: {count} occurances")
            
            
if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--montecarlo', type=bool, default=True)
    parser.add_argument('--sample_size', type=int, default=500000)
    parser.add_argument('--cms_filename', type=str, default='raw_cms.h5')
    args = parser.parse_args()
    
    divisions = [0.3, 0.3, 0.2, 0.2]
    
    print("Creating file now:")
    if args.montecarlo == True: 
        print("Assuming making monte carlo dataset")
        make_montecarlo_dataset(args.sample_size, args.filename, divisions)
    else: 
        print("Assuming making raw cms dataset")
        make_raw_cms_dataset(args.cms_filename, args.filename)
    
    print("File Specs:")
    report_file_specs(args.filename, divisions)
    
    