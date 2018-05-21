class Solver(object):
    pass

solver = Solver()
#==============

from data.data_loader1 import get_train_data_batch, get_test_data_batch
from model.net1 import xxx_net

solver.get_train_data_batch = get_train_data_batch
solver.get_test_data_batch = get_test_data_batch
solver.net = xxx_net()

solver.base_lr = 0.001
solver.lr_policy = "StepLR"
# solver.lr_policy = "MultiStepLR"
# solver.lr_policy = "LambdaLR"
# solver.lr_policy = "ExponentialLR"
# solver.lr_policy = "CosineAnnealingLR"
# solver.lr_policy = "ReduceLROnPlateau"
solver.gamma = 0.1
solver.stepsize = 100000
solver.max_iter = 450000
solver.display = 20
solver.save_training_result = False
solver.training_result_filename = ""

solver.optimizer = "Adam"
solver.momentum = 0.9
solver.momentum2 = 0.999
solver.weight_decay = 0.0005
solver.msgrad = False

solver.test_iter = 1000
solver.test_interval = 1000
solver.save_test_result = False
solver.test_result_filename = ""

solver.train_batchsize = 64
solver.test_batchsize = 64

solver.snapshot = 10000
solver.snapshot_prefix = "../snapshot/solver1_"

solver.solver_mode = "GPU"

solver.modelweight_path = ""
solver.solverstate_path = ""

#==============
from train import trainer
trainer.train(solver)