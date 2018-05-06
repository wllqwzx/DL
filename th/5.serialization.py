import torch
# 1. torch.load & torch.save

# file extension can be set to anything
# torch.save & load works for many data types:
# eg: int, float, str
#     array, dict, tuple
#     self defined object, 
torch.save(123, "test1.pkl")
torch.load("test1.pkl") #==> 123

torch.save([1,2,3,4], "test2.pkl")
torch.load("test2.pkl") #==> [1,2,3,4]

torch.save({"hello":123, "world":[1,2,3]}, "test3.pkl")
torch.load("test3.pkl") #==> {"hello":123, "world":[1,2,3]

a = torch.Tensor([1,2,3,4])
torch.save(a, "test4.pkl")
torch.load("test4.pkl") #==> tensor([ 1.,  2.,  3.,  4.])

# save and load a torch.nn.Module
net = torch.nn.Linear(100,10)
torch.save(net, "model.pkl")
net = torch.load("model.pkl")
net #==> Linear(in_features=100, out_features=10, bias=True)

# save and load a optimizer state
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
torch.save(optimizer, "optimizer.pkl")
optimizer = torch.load("optimizer.pkl")

#======
# we can use another way to save and load data for model and optimizer
# which don't save whole variable, but a dictionary
# this make it convenient for the saved weight to be used by other DL frameworks

net = torch.nn.Linear(10,10)
net.state_dict()
# {'bias': ..., 'weight': ...}

net2 = torch.nn.Linear(10,10)
net.load_state_dict(net2.state_dict())
# now net and net2 have the same weight ans bias value

# save model
torch.save(net.state_dict(), "model_state_dict.pkl")

# load model
net.load_state_dict(torch.load("model_state_dict.pkl"))


optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# save optimizer
torch.save(optimizer.state_dict(), "optimizer_state_duct.pkl")
# load optimizer
optimizer.load_state_dict(torch.load("optimizer_state_duct.pkl"))



#====== save more things
# exceot for model weight and optimizer state,
# sometimes we want to save more things, such as
# number of iters, number of batches and so on, we can put them
# in to a dict and save the whole dict

state = {
    'epoch': 10,
    'iters': 10000,
    'optimizer_state_dict': optimizer.state_dict()
}
torch.save(state, "state.pkl")

state = torch.load("state.pkl")
epoch = state['epoch']
iters = state['iters']
optimizer.load_state_dict(state['optimizer_state_dict'])
