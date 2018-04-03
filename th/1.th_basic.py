import torch as th
import numpy as np
from torch.autograd import Variable
'''
Tensor in pytorch is very different from Tensor in tensorflow.
In pytorch, Tensor is just a wrap of ndarray. Tensor is similar to numpy but can run on GPUs.
Like numpy arrays, PyTorch Tensors do not know anything about deep learning or computational graphs 
or gradients; they are a generic tool for scientific computing.

Variable in pytorch is more like Tensor in tensorflow.
In tensorflow, Tensor has 3 main forms: Variable(trainable/untrainable), constant, placeholder
'''

#=== 1. tensor type
'''
torch.FloatTensor   :   float32
torch.DoubleTensor  :   float64
torch.HalfTensor    :   float16
torch.ByteTensor    :   int8 (unsigned)
torch.CharTensor    :   int8 (signed)
torch.ShortTensor   :   int16
torch.IntTensor     :   int32
torch.LongTensor    :   int64
'''



#=== 2. numpy interfce 
# tensor -> ndarray
x = th.FloatTensor(3,3)
x.numpy()

# ndarray -> tensor
y = th.from_numpy(np.random.rand(5, 3))

#!!! type of tensor and numpy keeps same





#=== 3. tensor alignment
# 1. shape
x = th.FloatTensor(5, 3, 2)
x.size()    # torch.Size([5, 3, 2])

# 2. reshape
# use .view, do not use .resize
x = x.view([2,3,-1])
x.size()    # torch.Size([2, 3, 5])

# 3. transpose
# two axis
x = x.transpose(1,2)
x.size()    # torch.Size([2, 5, 3])

# all axis
x = x.permute(2,0,1)
x.size()    # torch.Size([3, 2, 5])

# 4. insert new axis
x = x.unsqueeze(1)
x.size()    # torch.Size([3, 1, 2, 5])

# 4_1. remove redundant axis(axis with length 1)
xx = Variable(th.Tensor([2,3,4,5,6,7,5,4]))
xx = xx.resize(2,1,2,1,2,1,1)   # torch.Size([2, 1, 2, 1, 2, 1, 1])
xx = xx.squeeze()   # torch.Size([2, 2, 2])

# 5. repeat
x = x.expand(3,10,2,5)
x.size()    # torch.Size([3,10,2,5])
#!!! expand can only repeat axis with size of 1
# eg: x = x.expand(3,1,20,5) => error!

# 6. concate
x = th.cat([x,x,x],1)
x.size()    # torch.Size([3, 30, 2, 5])

# 7. dot multiplication
x = Variable(torch.Tensor(5, 3))
y = Variable(torch.Tensor(3, 5))
print(th.mm(x,y))   # torch.Size([5, 5])

# 8. pair-wise multiplication
print(th.mul(x,x))  # torch.Size([5, 3])
# or just: x*x






#=== 4. init a Tensor with a distribution
#! Tensor's method end with "_" means this in_place operation
x = th.FloatTensor(5, 3, 2)
x.normal_(mean=0, std=1)
x.bernoulli_()
x.cauchy_(median=0, sigma=1)






#=== 5. max and argmax
# pytorch.max return both max value and its index, so call th.max for both max and argmax 
xx = Variable(th.Tensor(2,3,4,5,6)) # torch.Size([2, 3, 4, 5, 6])
val, index = th.max(xx, dim=4)
val.size()      # torch.Size([2, 3, 4, 5])
index.size()    # torch.Size([2, 3, 4, 5])

