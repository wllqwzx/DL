'''
After 0.4.0 Variable is deprecated, torch.Tensor itself can compute gradient.
'''
import torch


#========== compute gradient
x = torch.Tensor([1,2,3,4])
print(x.requires_grad)  #=> False

x.requires_grad = True
print(x.requires_grad)  #=> True

z = x.sum()
z.backward()
x.grad  #=> tensor([ 1.,  1.,  1.,  1.])


#========== Leaf node and intermediate node
'''
gradient can only be computed for leaf node,
intermediate node's .grad will always return None
even if its .requires_grad is True
'''
x1 = torch.randn((2,2), requires_grad=True) # leaf node
x2 = x1.cuda()  # intermediate node
z = (x2**2).sum()  # intermediate node
z.backward()
print(x1.grad)  # tensor([[ 0.2523,  1.2830], [-2.7661, -0.1376]])
print(x2.grad)  # None




#========== update tensor(requires_grad = True) value
x = torch.Tensor([1,2,3,4])
x.requires_grad = True

x.grad = None
z = x.sum()
z.backward()

# Wrong!!!, x will become an interminate node. so use inplace operation
x = x - 0.01*x.grad 

# Wrong!!!, sub is not an inplace operation, use .sub_
x.sub(0.01*x.grad)

# Wrong!!!, pytorch do not permite a leaf Tensor that requires grad 
# has been used in an in-place operation.
x.sub_(0.01*x.grad)

# This is OK
x.data.sub_(0.01*x.data.grad)



#========== an full example
x = torch.Tensor([1,2,3,4])
x.requires_grad = True

for i in range(10):
    # set grad to None before execute backward, otherwise
    # grad will accumulate(previous grad will sum with next grad)
    x.grad = None
    z = x.sum()
    z.backward()
    #x = x - 0.01*x.grad #!!! x becomes interminate mode
    x.data.sub_(0.01*x.grad)
print(x)