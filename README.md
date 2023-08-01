Dataset Descriptions: 
    - background_IDs_-1.npz : full labels for Delphes dataset 
    - bsm_datasets_-1.npz   : full anomaly dataset
    - datasets_-1.npz       : full features for Delphes dataset (sample_size, 19, 3, 1)
    - large_divisions.npz   : og 0.3, 0.3, 0.2, 0.2 split using old zscore preprocessing
    - preprocessed_cms.npz  : og cms using old zscore preprocessing 
    - raw_cms.h5            : raw cms data 
    - max_pt_scaling.npz    : first run max pt scaling with 0.3, 0.3, 0.2, 0.2 split
    - zscore.npz            : first run zscore (reduce mean dim 1, divide by constant std) with 0.3, 0.3, 0.2, 0.2 split
    - zscore_1.npz          : second run zscore, divide by constant mean and std 
    
Encoder Weights Descriptions: 
    - encoder_weights.h5    : original encoder weights with wrong z score preprocessing
    - max_pt_scaling.h5     : first run max pt scaling. LD 8, E 10, BS 1082, LR 0.05, T 0.07
    - max_pt_scaling_2.h5   : second run max pt scaling. LD 8, E 10, BS 541, LR 0.05, T 0.07
    - max_pt_scaling_3.h5   : third run max pt scaling. LD 8, E 10, BS 1082, LR 0.005, T 0.07
    - max_pt_scaling_4.h5   : fourth run max pt scaling. LD 8, E 10, BS 100, LR 0.005, T 0.07
    - max_pt_scaling_5.h5   : fifth run max pt scaling. LD 8, E 20, BS 100, LR 0.001, T 0.7
    - zscore_0.h5           : first run szcore. LD 8, E 10, BS, 1028, LR 0.05, T 0.07
    - 
