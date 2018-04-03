import torch as th
from torch.autograd import Variable
import numpy as np

#====== OOP API
'''
th.nn.Linear
th.nn.Conv2d
th.nn.ConvTranspose2d
th.nn.MaxPool2d
th.nn.LSTM
th.nn.GRU

th.nn.BatchNorm2d
th.nn.Dropout2d

th.nn.Sigmoid
th.nn.ReLU
th.nn.Tanh
th.nn.ELU

th.nn.MSELoss
th.nn.CrossEntropyLoss
'''

#=== Linear:
linear = th.nn.Linear(in_features=10, out_features=20, bias=True)
x = Variable(th.randn(32, 10))
out = linear(x)


#=== Conv + BatchNorm + Pooling
x = Variable(th.randn(10, 3, 28, 28))
conv = th.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=True)
bn = th.nn.BatchNorm2d(num_features=32)
pool = th.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
out = pool(bn(conv(x)))




#====== functional API
# the layers in scope th.nn are OOP style, we need first create and then use them.
# In th.nn.functional scope there are functional programming style, we can use them directly
import torch.nn.functional as F

# example
x = Variable(th.randn(10, 3, 28, 28))
filters = Variable(th.randn(32, 3, 3, 3))
conv_out = F.relu(F.dropout(F.conv2d(input=x, weight=filters, padding=1), p=0.5, training=True))

print('Conv output size : ', conv_out.size())   # torch.Size([10, 32, 28, 28])




#====== weight initialization
#!!! note that all off-the-shelf NNs in pytorch has already been inited while createing by default.
'''
th.nn.init.uniform
th.nn.init.xavier_normal
th.nn.init.xavier_uniform
th.nn.init.kaiming_normal
th.nn.init.kaiming_uniform
'''

# example:
conv_layer = th.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1)
for k,v in conv_layer.named_parameters():
    if k == 'weight':
        th.nn.init.kaiming_normal(v)    #!!! initilizer is in_place opeartion

