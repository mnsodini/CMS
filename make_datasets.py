print("Importing from 'make_datasets.py'")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import h5py
import numpy as np
import data_preprocessing
from argparse import ArgumentParser
import tensorflow as tf

def make_montecarlo_dataset(sample_size, new_filename, divisions, normalization_type): 
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

    # Extract sample_size samples from relevant files
    np.random.shuffle(train_ix) 
    x_train = data['x_train'][train_ix]
    x_test  = data['x_test'][test_ix]
    id_train = labels['background_ID_train'][train_ix]
    id_test = labels['background_ID_test'][test_ix]
    
    if normalization_type == 'max_pt':
        # Normalizes train and testing features by dividing by max pT. Saves weights in 'configs.py' file 
        data_preprocessing.save_normalization_weights(x_train, new_filename)
        x_train = data_preprocessing.maxPT_preprocess(x_train, new_filename)
        x_test = data_preprocessing.maxPT_preprocess(x_test, new_filename)

    elif normalization_type == 'zscore': 
        # Normalizes train and testing features by x' = (x - μ) / σ, where μ, σ are predetermined constants
        x_train = data_preprocessing.zscore_preprocess(x_train)
        x_test = data_preprocessing.zscore_preprocess(x_test)

    # Create and save new .npz with extracted features. Reports success 
    new_dataset = {'x_train': x_train, 'x_test': x_test, 'labels_train': id_train, 'labels_test': id_test}
    if not os.path.exists('Data'): os.makedirs('Data')
    file_path = os.path.join('Data', new_filename)
    np.savez(file_path, **new_dataset)
    print(f"{file_path} successfully saved")

def make_raw_cms_dataset(delphes_filter, new_filename, training_filename, normalization_type): 
    '''
    Given raw CMS, converts to npz file with appropriate transofmrations: 
    '''
    raw_cms_file = h5py.File('Data/raw_cms.h5', 'r')
    dataset_np = np.array(raw_cms_file['full_data_cyl'])

    if delphes_filter: 
        delphes_filter = raw_cms_file['L1_SingleMu22']
        filter_np  = np.array(delphes_filter)
        dataset_np = dataset_np[filter_np]
    
    # Reordering and reshaping 
    dataset_np = data_preprocessing.transform_raw_cms(dataset_np)
    
    # Convert from computer int embedding to meaningful float rep
    dataset_np = data_preprocessing.convert_to_float(dataset_np)
    
    # Phi Shift so range from [0, 2π] to [-π, π]
    dataset_np = tf.reshape(dataset_np, (-1, 19, 3, 1))
    dataset_np = data_preprocessing.phi_shift(dataset_np)
    
    # Either max_pT or zscore normalization. Reshapes/cast so shapes compatable 
    dataset_np = tf.reshape(dataset_np, (-1, 19, 3, 1))
    dataset_np = tf.cast(dataset_np, dtype=tf.float32)
    
    if normalization_type == 'max_pt':
        # Normalizes features by dividing by max pT. Uses training_filename to pull presaved max_pT weight
        dataset_np = data_preprocessing.maxPT_preprocess(dataset_np, training_filename)
    elif normalization_type == 'zscore': 
        # Normalizes features by x' = (x - μ) / σ, where μ, σ are predetermined constants
        dataset_np = data_preprocessing.zscore_preprocess(dataset_np)
    
    # Saves files and reports sucess 
    if not os.path.exists('Data'): os.makedirs('Data')
    subfolder_path = os.path.join('Data', new_filename)
    np.savez(subfolder_path, dataset=dataset_np)
    raw_cms_file.close()
    print(f"{subfolder_path} successfully saved")
    
    
def report_file_specs(filename, divisions): 
    '''
    Reports file specs: keys, shape pairs. If divisions, also reports number of samples from each label represented 
    in dataset 
    '''
    data = np.load('Data/' + filename, mmap_mode='r')
    for key in data.keys(): print(f"Key: '{key}' Shape: '{data[key].shape}'")

    if divisions != []: # prints frequency of each label
        label_counts = data['labels_train'].astype(int)
        label_counts = np.bincount(label_counts)
        for label, count in enumerate(label_counts): 
            print(f"Label {label}: {count} occurances")
            
            
if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()
    parser.add_argument('--new_filename', type=str, required=True)
    parser.add_argument('--training_filename', type=str, default='max_pt_scaling.npz')
    parser.add_argument('--delphes', type=bool, default=False)
    parser.add_argument('--sample_size', type=int, default=500000)
    parser.add_argument('--delphes_filter', type=bool, default=False)
    parser.add_argument('--normalization_type', type=str, default='max_pt')
    args = parser.parse_args()
    
    divisions = []
#     divisions = [0.30, 0.30, 0.20, 0.20]
    
    print("Creating file now:")
    if args.delphes == True: 
        print("Assuming making a Delphes Data Subset:")
        make_montecarlo_dataset(args.sample_size, args.filename, divisions, args.normalization_type)
    else: 
        print("Assuming making a Raw CMS Dataset:")
        make_raw_cms_dataset(args.delphes_filter, args.new_filename, args.training_filename, args.normalization_type)
    
    print("File Specs:")
    report_file_specs(args.new_filename, divisions)
    
    
