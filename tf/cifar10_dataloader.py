import tensorflow as tf
import numpy as np
import torchvision
import torchvision.transforms as transforms

__trainset = torchvision.datasets.CIFAR10(root='/data', train=True, download=True)

__testset = torchvision.datasets.CIFAR10(root='/data', train=False, download=True)


__X_train = __trainset.train_data / 255 # [50000, 3, 32, 32]
__Y_train = __trainset.train_labels     # [50000]

X_test = __testset.test_data / 255      # [10000, 3, 32, 32]
Y_test = __testset.test_labels          # [10000]

def private_public_split(private_ratio=0.5, seed=42):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(__X_train))
    private_set_size = int(private_ratio*len(__X_train))
    private_indices  = shuffled_indices[:private_set_size]
    public_indices   = shuffled_indices[private_set_size:]
    X_private_train = __X_train[private_indices]
    Y_private_train = __Y_train[private_indices]
    X_public_train  = __X_train[public_indices]
    Y_public_train  = __Y_train[public_indices] # do not need to return
    return X_private_train, Y_private_train, X_public_train, Y_public_train


X_private_train, Y_private_train, X_public_train, Y_public_train = private_public_split(private_ratio=0.5, seed=42)

def get_batch(data, batch, batch_size):
    length = len(data)
    start = batch*batch_size
    end = (batch+1)*batch_size
    if end > length:
        end = length
    return data[start:end]
