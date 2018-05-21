from model.net2 import xxx_net
from data.data_loader2 import get_train_data_batch, get_test_data_batch

net = xxx_net()

base_lr = 0.01
lr_policy = "step"
stepsize = 100000
max_iter = 450000
display = 20

optimizer = "Adam"
gamma = 0.1
momentum = 0.9
momentum2 = 0.999
weight_decay = 0.0005

test_iter = 1000
test_interval = 1000

snapshot = 10000
snapshot_prefix = "../snapshot/solver2_"

solver_mode = "GPU"