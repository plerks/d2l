import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB = dict()
DATA_URL = 'https://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):
    '''下载一个DATA_HUB中的文件, 返回本地文件名'''
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname) # f 所在目录
    data_dir, ext = os.path.splitext(fname) # 解压后目录，原扩展名
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    for name in DATA_HUB:
        download(name)

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]) # DataFrame.iloc[行选择, 列选择]，iloc 表示按下标索引
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) # [1:-1) 去掉了最后的那行 SalePrice

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index # 选出所有数值列
# all_features[numeric_features] 是一个 DataFrame
all_features[numeric_features] = all_features[numeric_features].apply( # apply 默认按列处理，不改原值，返回新对象
    lambda x: (x - x.mean()) / x.std()
)

# 此时已经标准化，把缺失值变 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 对所有类别特征做独热编码，不能按类别给个编号的原因在于会让模型认为有大小关系。dummy_na=True 表示将缺失值 NaN 也作为一列独立类型
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0] # 不包含表头
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
# print(train_features.shape) # 1460 个样本，331 个特征
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    '''
    nn.Linear(in_features, 1) 定义的其实不是：
        331个数 -> 1个数
    而是：
        任意个样本 × 331个特征 -> 任意个样本 × 1个输出
    PyTorch 默认把输入的最后一维看作特征维，其前面的维度都当作 batch 维度。
    '''
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

def log_rmse(net, features, labels): # 对数均方根误差
    clipped_preds = torch.clamp(net(features), 1, float('inf')) # 把数值变到 [1, inf] 之间
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels))) # 先取 log 再用 loss() ，这样就是对数均方根误差
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # weight_decay 为权重衰减的常数
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y) # 平均损失
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None: # 这个函数的使用，test_ls 是验证集或者测试集的误差
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    '''k: 折数, i: 当前第几折作为验证集'''
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx] # 一块
        if j == i:
            X_valid, y_valid = X_part, y_part # 验证集
        elif X_train is None:
            X_train, y_train = X_part, y_part # 测试集
        else:
            X_train = torch.cat([X_train, X_part], 0) # torch.cat 要求传进去的每个元素都必须是 Tensor，所以要两个分支
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1] # 加上最后一轮的损失
        valid_l_sum += valid_ls[-1]
        if i == 0: # 只画了第一折的损失变化曲线
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            d2l.plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs,
                   lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    d2l.plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    print(f'test_features.shape: {test_features.shape}')
    preds = net(test_features).detach().numpy()
    print(f'preds.shape: {preds.shape}')
    test_data['SalePrice'] = pd.Series(preds.flatten())
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('chapter4/submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)