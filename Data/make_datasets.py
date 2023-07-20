print('da imports')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import h5py
import numpy as np
import data_preprocessing
from argparse import ArgumentParser

def make_montecarlo_dataset(sample_size, filename, divisions): 
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
    x_train = data_preprocessing.zscore_preprocess(data['x_train'][train_ix])
    x_test  = data_preprocessing.zscore_preprocess(data['x_test'][test_ix])
    id_train = labels['background_ID_train'][train_ix]
    id_test = labels['background_ID_test'][test_ix]

    # Create and save new .npz with extracted features 
    new_dataset = {'x_train': x_train, 'x_test': x_test, 'labels_train': id_train, 'labels_test': id_test}
    print(f"{filename} successfully saved")
    np.savez(filename, **new_dataset)
    
    
def make_raw_cms_dataset(cms_filename, cms_dataname, new_filename): 
    '''
    Given raw CMS, converts to npz file with appropriate transofmrations: 
    Reordering and clipping -> integer representation to float -> phi wrapping
    '''
    raw_cms_file = h5py.File(cms_filename, 'r')
    raw_data = raw_cms_file[cms_dataname]
    dataset_np = np.array(raw_data)
    dataset_np = data_preprocessing.transform_raw_cms(dataset_np)
    dataset_np = data_preprocessing.convert_to_float(dataset_np)
    dataset_np = data_preprocessing.phi_shift(dataset_np)
    np.savez(new_filename, dataset=dataset_np)
    raw_cms_file.close()
    print("datset converted and saved as NPZ file")
    
    
def report_file_specs(filename, divisions): 
    data = np.load(filename, mmap_mode='r')
    for key in data.keys(): print(f"Key: '{key}' Shape: '{data[key].shape}'")

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
    parser.add_argument('--cms_dataname', type=str, default='full_data_cyl')
    args = parser.parse_args()
    
    divisions = [0.30, 0.30, 0.20, 0.20]
    
    print("Creating file now:")
    if args.montecarlo == True: make_new_dataset(args.sample_size, args.filename, divisions)
    else: h5_to_preprocessed_npz(args.cms_filename, args.cms_dataname, args.filename)
    
    print("File Specs:")
    report_file_specs(args.filename, divisions)
    
    