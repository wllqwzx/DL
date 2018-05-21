import os
import sys

import numpy as np
import torch as th
from tqdm import tqdm

def train(solver):
    net = solver.net

    #====== create optimizer
    if solver.optimizer == 'Adam':
        beta1 = solver.momentum
        beta2 = solver.momentum2
        optimizer = th.optim.Adam(net.parameters(), 
                                lr=solver.base_lr, 
                                betas=(beta1, beta2),
                                weight_decay=solver.weight_decay)
    elif solver.optimizer == 'SGD':
        optimizer = th.optim.SGD(net.parameters(),
                                lr=solver.base_lr,
                                momentum=solver.momentum,
                                weight_decay=solver.weight_decay)
    else:
        print("TODO! only support Adam and SGD")
        return

    #====== restore model weight
    if "modelWeight_path" in solver.__dict__:
        if solver.modelweight_path != "":
            if not os.path.exists(solver.modelweight_path):
                print("Error: modelweight file: "+solver.modelweight_path+" does not exist!")
                return
            else:
                modelweight = th.load(solver.modelweight_path)
                net.load_state_dict(modelweight)

    #====== restore optimizer state 
    init_iters = 1
    if "solverState_path" in solver.__dict__:
        if solver.solverstate_path != "":
            if not os.path.exists(solver.solverstate_path):
                print("Error: solverstate file: "+solver.solverstate_path+" does not exist!")
                return
            else:
                solverstate = th.load(solver.solverstate_path)
                optimizer.load_state_dict(solverstate['optimizer_state_dict'])
                init_iters = solverstate['iters']

    #====== set solver mode
    if solver.solver_mode == "GPU":
        net = net.cuda()
    else:
        print("TODO! only supports GPU mode")
        return


    #====== start training
    iters = 0
    net.train()
    for iters in tqdm(range(init_iters, solver.max_iter+1)):
        # TODO: traininig the net 
        train_data = solver.get_train_data_batch(solver.train_batchsize)
        train_net_tops = net.train_forward(*train_data)
        for k in train_net_tops:
            if k == 'loss':
                optimizer.zero_grad()
                train_net_tops['loss'].backward()
                optimizer.step()
            else:
                pass

        if iters % solver.stepsize == 0:
            # TODO: change learning rate
            pass
        
        if iters % solver.display == 0:
            # TODO: display intermedinate result
            pass

        if iters % solver.test_iter == 0:
            # TODO: report test result
            pass

        if iters % solver.snapshot == 0:
            # TODO: save weight and optimizer state
            pass

        if iters >= solver.max_iter:
            print(iters)
            # TODO: display intermedinate result
            # TODO: report test result
            # TODO: save weight and optimizer state
            break

def __display_result():
    pass

def __save_snapshot():
    pass

