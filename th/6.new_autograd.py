'''
After 0.4.0 Variable is deprecated, torch.Tensor itself can compute gradient.
'''
import torch
'''
    1. compute gradient
    2. Leaf node and intermediate node
    3. update tensor(requires_grad = True) value
    4. an full example
    5. use .detach to stop backpropagation for this node
    6. model.train() and model.eval() and torch.no_grad()
'''

#========== 1. compute gradient
x = torch.Tensor([1,2,3,4])
print(x.requires_grad)  #=> False

x.requires_grad = True
print(x.requires_grad)  #=> True

z = x.sum()
z.backward()
x.grad  #=> tensor([ 1.,  1.,  1.,  1.])


#========== 2. Leaf node and intermediate node
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




#========== 3. update tensor(requires_grad = True) value
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
x.data.sub_(0.01*x.grad.data)



#========== 4. an full example
x = torch.Tensor([1,2,3,4])
x.requires_grad = True

for i in range(10):
    # set grad to None before execute backward, otherwise
    # grad will accumulate(previous grad will sum with next grad)
    x.grad = None
    z = x.sum()
    z.backward()
    #x = x - 0.01*x.grad #!!! x becomes interminate node
    x.data.sub_(0.01*x.grad.data)
print(x)



#========== 5. use .detach to stop backpropagation for this node
x = torch.Tensor([1,2,3,4])
x.requires_grad = True

y1 = (x*x).sum()        # gradient pass throught z -> y1 -> x
y2 = x.detach().mean()  # gradient will not pass throught y2 to x

z = y1*y2
z.backward()
print(x.grad)



#========== 6. model.train() and model.eval() and torch.no_grad()
'''
Note that model.train() and model.eval() just set a state(property) of the model,
it is used only for dropout, bn, etc layer to decide its computing behaivor.
IT DOSE NOT AFFECT the reqiures_grad of any tensor, so in eval mode, you need to
set all tensor's requires_grad to False.

pytorch 0.4.0 provide torch.no_grad() which can do there for us:
    1. reqiures_grad of all interminate node create inside will forced to be False automatically
    2. call .backward() inside it will raise a runtime error

original doc:
    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call Tensor.backward(). It will reduce memory
    consumption for computations that would otherwise have requires_grad=True.
    In this mode, the result of every computation will have
    requires_grad=False, even when the inputs have requires_grad=True.
'''
x = torch.Tensor([1,2,3,4])
x.requires_grad = True
with torch.no_grad():
    z = x.sum()
    #z.backward() # this will cause a runtime error
    z.reqiures_grad = True
    print(z.requires_grad)  # still equals to False