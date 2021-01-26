import torch
import numpy as np

def torch_mae(y_pred, y_true):
    loss = (torch.abs(y_pred - y_true)).mean()
    return loss  

def numpy_mape(y_pred, y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

def numpy_smape(y_pred, y_true):
    return 100/len(y_pred) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))