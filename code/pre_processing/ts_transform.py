import numpy as np
import pandas as pd 
from utils import progressBar

def sliding_window(x, window_size, stride):
    
    windows = []
    for idx in range(0, len(x), stride):
        windows.append(x[np.newaxis, idx:idx+window_size,:])
        #print(windows[0].shape)
    return np.vstack(windows)


def scale_datasets(standarize = True, normalize = True, *datasets):

    train_dataset, validation_dataset, test_dataset = datasets

    if normalize:
        train_dataset.compute_min_max_params()
        max_list, min_list = train_dataset.get_min_max_params()
        validation_dataset.set_min_max_params(max_list, min_list)
        test_dataset.set_min_max_params(max_list, min_list)

        train_dataset.normalize()
        validation_dataset.normalize()
        test_dataset.normalize()
        
    if standarize:
        train_dataset.compute_standard_params()
        mean, std = train_dataset.get_standard_params()
        validation_dataset.set_standard_params(mean, std)
        test_dataset.set_standard_params(mean, std)

        train_dataset.standarize()
        validation_dataset.standarize()
        test_dataset.standarize()

def sample_to_sktime(sample):
    
    #inspired by tslearn https://github.com/tslearn-team/tslearn/blob/775dadd/tslearn/utils.py#L867-L939
    X_ = sample
    X_pd = pd.DataFrame(dtype=np.float32)
    for dim in range(X_.shape[2]):
        X_pd['dim_' + str(dim)] = [pd.Series(data=Xi[:Xi.shape[0], dim])
                                   for Xi in X_]#
    
    return X_pd

def dataset_to_sktime(dataset):

    sktime_dataset = []

    print("Transforming dataset to sktime format...")

    for i, sample in enumerate(dataset.x):

        progressBar(i, dataset.x.shape[0],40 )
        sktime_dataset.append(sample_to_sktime(sample[np.newaxis,:]))

    return sktime_dataset


def apply_rocket_kernels(sktime_dataset, rocket):

    data_features = []

    for i, sample in enumerate(sktime_dataset):
        
        progressBar(i, len(sktime_dataset), 40)
        data_features.append(rocket.transform(sample))

    data_features = np.concatenate(data_features, axis=0)
    
    return data_features

def split_idx_50_50(domain_idx):

    domain_idx = np.array(domain_idx)
    n_domains = np.max(domain_idx)+1
    print(n_domains)

    domain_change = [0]
    for d in range(n_domains):

        domain_change.append(np.sum(domain_idx==d))

    domain_change = np.cumsum(domain_change)
    print(domain_change)
    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(len(domain_change)-1):

        pos1 = domain_change[i]
        pos2 = (domain_change[i]+ int(0.9*(domain_change[i+1]-domain_change[i])/2))
       # pos22 = (domain_change[i]+ int(0.9*(domain_change[i+1]-domain_change[i])/2))
        pos3 = (domain_change[i]+domain_change[i+1])/2
        pos4 = domain_change[i+1]
        
        print(pos1)
        train_idx.append(np.arange(pos1, pos2).astype(int))
        val_idx.append(np.arange(pos2, pos3).astype(int))
        test_idx.append(np.arange(pos3, pos4).astype(int))

    
    return train_idx, val_idx, test_idx