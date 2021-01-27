import pandas as pd 
import numpy as np
import copy



class TSDataset(object):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    
    def __init__(self, 
                data,
                window_size,
                task_size,
                stride = 1,
                mode = "meta-learning",
                file_counter = 0,
                filter = False):
        
        x_list = []
        y_list = []
        
        for i in range(0, len(data)-window_size, stride):             
            x_list.append(data[i:i+window_size, :-1][np.newaxis,:])
            y_list.append(data[i+window_size,-1])
        
        self.x = np.vstack(x_list)
        self.y = np.vstack(y_list)

        if(mode == "meta-learning"):

            x_list = []
            y_list = []
            
            for i in range(0, len(self.x)-task_size, task_size+window_size): #no-overlapping
                x_list.append(self.x[i:i+task_size,...][np.newaxis,...])
                y_list.append(self.y[i:i+task_size,...][np.newaxis,...])

            print(x_list[0].shape)
            self.x = np.concatenate(x_list, axis=0)
            self.y = np.concatenate(y_list, axis=0)
        
        self.task_size = task_size
        self.dim = self.x.shape[-1]
        self.mode = mode

        if filter:   
            self.filter_data()


        self.file_idx = [file_counter]*self.x.shape[0]
        
        print("x shape:", self.x.shape)
        print("y shape:", self.y.shape)
 

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]

    def __add__(self, other):
        object_copy = copy.deepcopy(self)
        object_copy.x = np.concatenate([object_copy.x, other.x], axis=0)
        object_copy.y = np.concatenate([object_copy.y, other.y], axis=0)
        
        object_copy.dim = object_copy.dim 
        object_copy.file_idx = object_copy.file_idx + other.file_idx
        
        return object_copy
    
    def compute_min_max_params(self):

        self.max_y = np.max(self.y)
        self.max_list = [np.max(self.x.reshape(-1, self.dim), axis=0), self.max_y]

        self.min_y = np.min(self.y)
        self.min_list = [np.min(self.x.reshape(-1, self.dim), axis=0), self.min_y]

    def set_min_max_params(self, max_list, min_list):

        self.max_list =  max_list
        self.min_list = min_list

    def get_min_max_params(self):

        return self.max_list, self.min_list


    def normalize(self):
        
        max_min_diff_x = np.array(self.max_list[0])-np.array(self.min_list[0])
        max_min_diff_x[max_min_diff_x==0.] = 1.
        max_min_diff_y = self.max_list[-1]-self.min_list[-1]
        max_min_diff_y = max_min_diff_y if max_min_diff_y !=0. else 1.

        self.x = np.divide(self.x - self.min_list[0],max_min_diff_x)
        self.y = (self.y - self.min_list[-1])/(max_min_diff_y)        
      
    def compute_standard_params(self):

        self.mean = np.mean(self.x.reshape(-1, self.dim), axis=0, keepdims=True)
        self.std = np.std(self.x.reshape(-1, self.dim), axis=0, keepdims=True)
        self.std[self.std==0.] = 1.

    def set_standard_params(self, mean, std):

        self.mean = mean
        self.std = std

    def get_standard_params(self):

        return self.mean, self.std

    def standarize(self):
        
        original_size = self.x.shape
        x_temp = self.x.reshape(-1, self.dim)
        self.x = ((x_temp - self.mean)/self.std).reshape(original_size)

    def filter_data(self):

        print("Filtering samples with invalid targets...")

        if(self.mode == "meta-learning"):
            where_to_delete = np.where(np.sum(self.y!=0, axis=1)<self.task_size*0.5)[0]

        else:
            w = 500
            conv =  np.convolve(self.y.reshape(-1), np.ones(w), 'same') / w
            where_to_delete = np.where(conv==0)[0]

        print("where to delete:", where_to_delete.shape)
        self.x = np.delete(self.x, where_to_delete, axis=0)
        self.y = np.delete(self.y, where_to_delete, axis=0)

class DomainTSDataset:
    
    def __init__(self, dataset = None, x=[], y=[], d=[]):
        
        
        if dataset is not None:

            self.x = dataset.x
            self.y = dataset.y
            self.d = dataset.file_idx

        else:

            self.x = np.array(x)
            self.y = np.array(y)
            self.d = np.array(d)


        
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.d[index]

    def __len__(self):
        return self.y.shape[0]

    def __add__(self, other):

        object_copy = copy.deepcopy(self)
        object_copy.x = np.concatenate([object_copy.x, other.x], axis=0)
        object_copy.y = np.concatenate([object_copy.y, other.y], axis=0)
        object_copy.d = np.concatenate([object_copy.d, other.d], axis=0)
        
        return object_copy

class SimpleDataset(object):

    def __init__(self, x, y):

        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]
