import yaml
import pandas as pd
import numpy as np
from torch.utils import data
import torch

def config_loader(path):
    with open(path, 'r') as stream:
        cfgs = yaml.safe_load(stream)  # 读取 YAML 文件内容并转换为字典
    return cfgs

def normalize_data(data):
    """
    Normalize the data column by column to the range [0, 1]. 
    Parameters:
    data: A 2D NumPy array (samples, features) containing the original data.
    Returns:
    normalized_data: The normalized data.
    min_vals: The minimum value of each column (1D array).
    max_vals: The maximum value of each column (1D array).
    """
    # Ensure the input is a two-dimensional array.
    if data.ndim == 1:
        data = data[:, np.newaxis]
    
    min_vals = np.min(data, axis=0)  # Minimum value of each column
    max_vals = np.max(data, axis=0)  # Maximum value of each column
    normalized_data = (data - min_vals) / (max_vals - min_vals + 1e-8)  # Broadcasting mechanism
    return normalized_data, min_vals, max_vals


def denormalize_data(normalized_data, min_vals, max_vals, cfgs):
    """
    Restores the normalized data column by column to its original range. 
    Parameters:
    normalized_data: The normalized data (samples, features)
    min_vals: The minimum value of each column (1D array)
    max_vals: The maximum value of each column (1D array)
    Returns:
    original_data: The restored data
    """
    max_vals = np.array(max_vals)
    min_vals = np.array(min_vals)
    label_cols = cfgs["label_cols"]
    if not isinstance(label_cols, list):
            label_cols = [label_cols]
    # Ensure the input is a two-dimensional array.
    if normalized_data.ndim == 1:
        normalized_data = normalized_data[:, np.newaxis]
    original_data = normalized_data * (max_vals[label_cols] - min_vals[label_cols] + 1e-8) + min_vals[label_cols]   
    return original_data

def read_process_data(cfgs):
    '''
    Returns:
    x (numpy.ndarray): Training data, with shape (number of samples, seq_len, len(input_cols)). 
    y (numpy.ndarray): Target data, with shape (number of samples, len(label_cols)).
    '''
    input_cols = cfgs["inputs_cols"]
    label_cols = cfgs["label_cols"]
    seq_len = cfgs["seq_len"]
    split_rate = cfgs["split_rate"]
    predict_data = cfgs["predict_data"]
    filepath = cfgs["dataroot"]
    try:
        data = pd.read_csv(filepath)
        data, min_val, max_val = normalize_data(data)

        if not isinstance(label_cols, list):
            label_cols = [label_cols]
        if any(col >= data.shape[1] for col in input_cols + label_cols): # Logically combine them, merging them into a single list.
            raise ValueError("指定的列索引超出csv文件的列范围")
        
        inputs = data.iloc[:, input_cols].values
        labels = data.iloc[:, label_cols].values 
        x, y = [], []
        for i in range(len(labels) - seq_len - predict_data + 1):
            x.append(inputs[i:i+seq_len, :])
            y.append(labels[i+seq_len:i+seq_len+predict_data, :])
        x, y = np.array(x), np.array(y)

        split_index = int(len(x) * split_rate)
        x_train, y_train = x[:split_index], y[:split_index]
        x_test, y_test = x[split_index:], y[split_index:]

        return x_train, y_train, x_test, y_test, min_val, max_val
    except Exception as e:
        print(f"处理csv文件时出错:{e}")
        return None, None, None, None, None, None
    
def makedataloader(x_train, y_train, x_test, y_test, cfgs):
    class LSTMdataset(data.Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels
        def __getitem__(self, index):
            input = self.inputs[index]
            label = self.labels[index]
            # Check if the input is a nested structure; if so, convert it to the correct numerical type.
            if isinstance(input, np.ndarray):
                input = input.astype(np.float32)  # Convert to float64 type
            if isinstance(label, np.ndarray):
                label = label.astype(np.float32)  
            # Convert the data to a Tensor.
            input = torch.tensor(input)  
            label = torch.tensor(label)  
            return input, label
        def __len__(self):
            return len(self.inputs)
    train_ds = LSTMdataset(x_train, y_train)
    test_ds = LSTMdataset(x_test, y_test)
    train_dl = data.DataLoader(
        train_ds,
        batch_size=cfgs["batch_size"],
        shuffle=False
    )
    test_dl = data.DataLoader(
        test_ds,
        batch_size=cfgs["batch_size"]
    )
    return train_dl, test_dl

def adjust_batch_first(inputs, labels, cfgs):
    if not cfgs["batch_first"]:
        # If `batch_first` is False, the dimension order needs to be adjusted.
        inputs = inputs.permute(1, 0, 2)  # (seq_len, batch_size, input_size)
        labels = labels.permute(1, 0, 2)  # (seq_len, batch_size, input_size)
    return inputs, labels