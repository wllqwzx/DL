import torch as th
import numpy as np
from torch.autograd import Variable
import torchvision

'''
In pytorch, Variable contains both data and its gradient.

Create a Variable require 2 parameters:
data            : the Tensor data type to init the value of Variable

requires_grad   : True/False, control whether the Variable is trainable, default value is False,
                  however the parameters of a model(NN) in pytorch requires grad in default.
                  Note: If there’s a single note in an subgraph that requires gradient, 
                  this subgraph will also require gradient. Conversely, only if all nodes 
                  don’t require gradient, the subgraph also won’t require it. Backward computation 
                  is never performed in the subgraphs, where all Variables(nodes) didn’t require gradients.

volatile        : Volatile should only set to be True for purely inference mode. If volatile is True, then
                  no intermediate state will be saved for compute gradient, thus is effecient for inference time.
                  Volatile has a higher priority than requires_grad, if one node in a graph is volatile(=True) 
                  then the whole graph is volatile, and also all of the requires_grad automaticly becomes False
                  Usually we need two input variable, one for training time and one for inference time.
                  Note that, once a graph is volatile, gradient can not be computed and calling .backward() will
                  raise an error!

'''

'''
Dynamic Graph:
An important thing to note is that the graph is recreated from scratch at every iteration, 
and this is exactly what allows for using arbitrary Python control flow statements, that 
can change the overall shape and size of the graph at every iteration. You don’t have to 
encode all possible paths before you launch the training - what you run is what you differentiate.
'''

'''
In-place operations are not recommand:
Supporting in-place operations in autograd is a hard matter, and we discourage their use in most cases. 
Autograd’s aggressive buffer freeing and reuse makes it very efficient and there are very few occasions 
when in-place operations actually lower memory usage by any significant amount. Unless you’re operating 
under heavy memory pressure, you might never need to use them.
'''

x = Variable(th.randn(5, 5))
y = Variable(th.randn(5, 5))
z = Variable(th.randn(5, 5), requires_grad=True)

a = x + y
print(a.requires_grad)  # => False

b = a + z
print(b.requires_grad)  # => True

#=====
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False     # freeze the parameters of the pretrained NN

# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
model.fc = th.nn.Linear(512, 100)

# Optimize only the classifier
optimizer = th.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)


#=====
regular_input = Variable(th.randn(10, 3, 224, 224))
volatile_input = Variable(th.randn(10, 3, 224, 224), volatile=True)
model = torchvision.models.resnet18(pretrained=True)

print(model(regular_input).requires_grad)   # => True
print(model(volatile_input).requires_grad)  # => False
print(model(volatile_input).volatile)       # => True


# compute gradient
#=====
x = Variable(torch.Tensor([5]), requires_grad=True)
y = Variable(torch.Tensor([3]), requires_grad=True)

# forword
z = x*y + x

# backword
z.backward()

dz_dx = x.grad   # => 4 
dz_dy = y.grad   # => 5

#=== or
x = Variable(torch.Tensor([5]), requires_grad=True)
y = Variable(torch.Tensor([3]), requires_grad=True)

z = x*y + x

dz_dx, dz_dy = th.autograd.grad(z, [x,y])

'''
About backword():
1. Calling .backward() clears the current computation graph.
2. Once .backward() is called, intermediate variables used in the construction of the graph are removed.
3. This is used implicitly to let PyTorch know when a new graph is to be built for a new minibatch. 
4. In some spacial cases, we may want to retain the graph after the backward pass, at that time
   we can use loss.backward(retain_graph=True). Note that when retain_graph is set to True. 

Note that: calling backward multiple times will accumulate gradients into .grad and NOT overwrite them.
           so, uauslly we need set all grads to zero before each calling of .backward()
'''



