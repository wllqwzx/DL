class Solver:pass
solver = Solver()
#==============

from data_loader.data_loader import get_train_data_batch, get_test_data_batch
from model.net import resnet

solver.image_size = 224

solver.get_train_data_batch = get_train_data_batch
solver.get_test_data_batch = get_test_data_batch
solver.net = resnet(layers=50)

solver.base_lr = 0.1
solver.lr_policy = "StepLR"
# solver.lr_policy = "MultiStepLR"
# solver.lr_policy = "LambdaLR"
# solver.lr_policy = "ExponentialLR"
# solver.lr_policy = "CosineAnnealingLR"
# solver.lr_policy = "ReduceLROnPlateau"
solver.gamma = 0.1
solver.stepsize = 30000
solver.max_iter = 200000
solver.display = 100
solver.save_training_result = False
solver.training_result_filename = ""

solver.optimizer = "SGD"
solver.momentum = 0.9
solver.momentum2 = 0.999
solver.weight_decay = 0.0005
solver.msgrad = False

solver.test_iter = 300
solver.test_interval = 1000
solver.save_test_result = False
solver.test_result_filename = ""

solver.train_batchsize = 64
solver.test_batchsize = 64

solver.snapshot = 10000
solver.snapshot_prefix = "./snapshot/solver_xxx_resnet50_sgd_"

solver.test_loss_file = "./snapshot/test_loss_solver_xxx.txt"
solver.train_loss_file = "./snapshot/train_loss_solver_xxx.txt"

solver.solver_mode = "GPU"

# training from a snapshot
solver.modelweight_path = ""
solver.solverstate_path = ""

#==============
from train import trainer
trainer.train(solver)


#==============
""" result:
validation loss:
validation accuracy:
best iter number:

"""
