import numpy as np
import pandas as pd
import torch as th

import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.MNIST(root="~/data/", train=True, transform=transforms.ToTensor, download=True)
test_dataset = torchvision.datasets.MNIST(root="~/data/", train=False, transform=transforms.ToTensor, download=True)

print("========== data set info: ==========")
print(train_dataset.train_data.numpy().shape)    # ndarray: [60000, 28, 28]
print(train_dataset.train_labels.numpy().shape)  # ndarray: [60000] :0~9
print(test_dataset.test_data.numpy().shape)      # ndarray: [10000, 28, 28]
print(test_dataset.test_labels.numpy().shape)    # ndarray: [10000] :0~9

# all data: ndarray
X_train = train_dataset.train_data.numpy().reshape([60000, 1, 28, 28])
y_train = train_dataset.train_labels.numpy()
X_test = test_dataset.test_data.numpy().reshape([10000, 1, 28, 28])
y_test = test_dataset.test_labels.numpy()


train_shuffled_indices = np.random.permutation(len(X_train))
train_current_index = 0
def get_train_data_batch(batch_size):
    """
    return tenor data for train_forward function.
    """
    global train_current_index
    global train_shuffled_indices
    if train_current_index+batch_size<len(X_train):
        indices = train_shuffled_indices[train_current_index:train_current_index+batch_size]
        X_batch = X_train[indices]
        Y_batch = y_train[indices]
        train_current_index = train_current_index + batch_size
    else:
        indices = train_shuffled_indices[train_current_index:]
        X_batch = X_train[indices]
        Y_batch = y_train[indices]
        train_current_index = 0
        np.random.seed(np.random.randint(0, 99999999))
        train_shuffled_indices = np.random.permutation(len(X_train))
    return th.Tensor(X_batch), th.LongTensor(Y_batch)



test_shuffled_indices = np.random.permutation(len(X_test))
test_current_index = 0
def get_test_data_batch(batch_size):
    """
    return tenor data for test_forward function.
    """
    global test_current_index
    global test_shuffled_indices
    if test_shuffled_indices+batch_size<len(X_test):
        indices = test_shuffled_indices[test_current_index:test_current_index+batch_size]
        X_batch = X_test[indices]
        Y_batch = y_test[indices]
        test_current_index = test_current_index + batch_size
    else:
        indices = test_shuffled_indices[test_current_index:]
        X_batch = X_test[indices]
        Y_batch = y_test[indices]
        test_current_index = 0
        np.random.seed(np.random.randint(0, 99999999))
        test_current_index = np.random.permutation(len(X_test))
    return th.Tensor(X_batch), th.LongTensor(Y_batch)
