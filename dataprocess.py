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
    将数据逐列归一化到 [0, 1]。
    参数:
        data: 二维 NumPy 数组 (samples, features)，原始数据
    返回:
        normalized_data: 归一化后的数据
        min_vals: 每列数据的最小值 (1D 数组)
        max_vals: 每列数据的最大值 (1D 数组)
    """
    # 保证输入是二维数组
    if data.ndim == 1:
        data = data[:, np.newaxis]
    
    min_vals = np.min(data, axis=0)  # 每列最小值
    max_vals = np.max(data, axis=0)  # 每列最大值
    normalized_data = (data - min_vals) / (max_vals - min_vals + 1e-8)  # 广播机制
    return normalized_data, min_vals, max_vals


def denormalize_data(normalized_data, min_vals, max_vals, cfgs):
    """
    将归一化后的数据逐列还原到原始范围。
    参数:
        normalized_data: 归一化后的数据 (samples, features)
        min_vals: 每列的最小值 (1D 数组)
        max_vals: 每列的最大值 (1D 数组)
    返回:
        original_data: 还原后的数据
    """
    max_vals = np.array(max_vals)
    min_vals = np.array(min_vals)
    label_cols = cfgs["label_cols"]
    if not isinstance(label_cols, list):
            label_cols = [label_cols]
    # 保证输入是二维数组
    if normalized_data.ndim == 1:
        normalized_data = normalized_data[:, np.newaxis]
    original_data = normalized_data * (max_vals[label_cols] - min_vals[label_cols] + 1e-8) + min_vals[label_cols]   
    return original_data

def read_process_data(cfgs):
    '''
    返回:
        x (numpy.ndarray): 训练数据，形状为 (样本数, seq_len, len(input_cols))。
        y (numpy.ndarray): 目标数据，形状为 (样本数, len(label_cols))。
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
        if any(col >= data.shape[1] for col in input_cols + label_cols): #逻辑合并， 合并成一个列表
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
            # 检查输入是否为嵌套结构，如果是，转换为正确的数值类型
            if isinstance(input, np.ndarray):
                input = input.astype(np.float32)  # 转换为 float64 类型
            if isinstance(label, np.ndarray):
                label = label.astype(np.float32)  
            # 将数据转换为 Tensor
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
        # 如果 batch_first 为 False，则需要调整维度顺序
        inputs = inputs.permute(1, 0, 2)  # (seq_len, batch_size, input_size)
        labels = labels.permute(1, 0, 2)  # (seq_len, batch_size, input_size)
    return inputs, labels