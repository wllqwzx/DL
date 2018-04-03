'''
Ref: http://pytorch.org/docs/0.3.1/notes/extending.html#extending-torch-autograd
----------

pytorch provide the basic primitive operations on Variable, we can use those operation
to build computing graph, and get the gradient automatically. But, sometimes, those
operatins may not satisfy our needs, for example: we need to implement a function that 
can't be composed of the existing primitive operations, then we need to implement them
ourselves, and we also need to define how to calculate gradient for that funtion.
Sometimes, we can implement our function with the default operations, but the function 
is complex, the compution may be slow, so we also implement it ourselves to accelerate
it, we can even call CUDA code in the forward and backward fnuction.

Here is an example:
'''

"""
Note: we only need to implement forward and backword function.
forward take inputs' value and return output's value, backward
take output's gradient w.r.t loss, and return inputs' gradients
w.r.t loss.

backward implemntation guide:

because: 
	d_loss/d_param1 = d_loss/d_func_out * d_func_out/d_param1
	d_loss/d_param2 = d_loss/d_func_out * d_func_out/d_param2

backward input and output:
input: d_loss/d_func_out
	compute: 
		d_func_out/d_param1
		d_func_out/d_param2
return: d_loss/d_func_out * d_func_out/d_param1, d_loss/d_func_out * d_func_out/d_param2

"""


import torch
from torch.autograd import Variable 
from torch.autograd import Function


class LinearFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):	#!!! type of input, weight ... all will be transformed to DoubleTensor
        '''
        print("ctx:",type(ctx))       #==> DoubleTensor
        print("input:",type(input))   #==> DoubleTensor
        print("weight:",type(weight)) #==> DoubleTensor
        '''
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output


    @staticmethod
    def backward(ctx, grad_output):	#!!! type of grad_output is Variable
        print("grad_output:", type(grad_output))
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None.
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output

        return grad_input, grad_weight, grad_bias

#==== use it
linear = LinearFunction.apply

#=== check whether the forward and backward meet each other
from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (Variable(torch.randn(20,20).double(), requires_grad=True), Variable(torch.randn(30,20).double(), requires_grad=True),)
test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
print(test)	#==> True






#===== another example
class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant): #!!! all input will be transformed into DoubleTensor
        '''
        print("tensor:", type(tensor))      #==> DoubleTensor
        print("constant:", type(constant))  #==> DoubleTensor
        '''
        # ctx is a context object that can be used to store information
        # for backward computation
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None

mul_constant = MulConstant.apply

input = (Variable(torch.randn(20,20).double(), requires_grad=True), 5)
test = gradcheck(mul_constant, input, eps=1e-6, atol=1e-4)
print(test)	#==> True
