import os
import sys

import numpy as np
import torch as th
from tqdm import tqdm


def adjust_learning_rate_stepLR(optimizer, step, init_lr, lr_decay_factor=0.1, lr_decay_step=10):
    """Sets the learning rate to the initial LR decayed by lr_decay_factor every lr_decay_step steps"""
    lr = init_lr * (lr_decay_factor ** (step // lr_decay_step))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    if "modelweight_path" in solver.__dict__:
        if solver.modelweight_path != "":
            if not os.path.exists(solver.modelweight_path):
                print("Error: modelweight file: " +
                      solver.modelweight_path + " does not exist!")
                return
            else:
                print(">>>>>>>>>> restore modelweight from: ",
                      solver.modelweight_path)
                modelweight = th.load(solver.modelweight_path)
                net.load_state_dict(modelweight)

    #====== restore optimizer state
    init_iters = 1
    if "solverstate_path" in solver.__dict__:
        if solver.solverstate_path != "":
            if not os.path.exists(solver.solverstate_path):
                print("Error: solverstate file: " +
                      solver.solverstate_path + " does not exist!")
                return
            else:
                print(">>>>>>>>>> restore modelweight from: ",
                      solver.solverstate_path)
                solverstate = th.load(solver.solverstate_path)
                optimizer.load_state_dict(solverstate['optimizer_state_dict'])
                
                # transform optimizer state to gpu type
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, th.Tensor):
                            state[k] = v.cuda()
                init_iters = solverstate['iters']
                adjust_learning_rate_stepLR(optimizer, init_iters,
                                            init_lr=solver.base_lr,
                                            lr_decay_factor=solver.gamma,
                                            lr_decay_step=solver.stepsize)

    #====== set solver mode and data parallel
    if solver.solver_mode == "GPU":
        # net = th.nn.DataParallel(net).cuda()
        net = net.cuda()
    else:
        print("TODO! only supports GPU mode")
        return

    #====== start training
    net.train()
    print(">>>>>>>>>> start training...")
    #for iters in tqdm(range(init_iters, solver.max_iter + 1)):
    for iters in range(init_iters, solver.max_iter + 1):
        # training the net
        train_data = solver.get_train_data_batch(solver.train_batchsize, solver.image_size)
        train_net_tops = net.forward(*train_data)
        for k in train_net_tops:
            if k == 'loss':
                optimizer.zero_grad()
                train_net_tops['loss'].backward()
                optimizer.step()

        # log train batch loss to file
        with open(solver.train_loss_file, "a") as f:
            f.write(str(iters)+" ")
            f.write(str(train_net_tops['loss'].cpu().detach().numpy().tolist()))
            f.write("\n")



        if iters % solver.stepsize == 0:
            # change learning rate
            adjust_learning_rate_stepLR(optimizer, iters,
                                        init_lr=solver.base_lr,
                                        lr_decay_factor=solver.gamma,
                                        lr_decay_step=solver.stepsize)

        if iters % solver.display == 0:
            # display intermedinate result
            for k in train_net_tops:
                print("[iters:", iters, "of", solver.max_iter, "] ", k + ": ", train_net_tops[k])

        # report test result
        if iters % solver.test_interval == 0:
            __display_test(net, solver, iters)

        # save weight and optimizer state
        if iters % solver.snapshot == 0:
            __save_snapshot(net, optimizer, iters, solver.snapshot_prefix)

        if iters >= solver.max_iter:
            print(iters)
            # display intermedinate result
            for k in train_net_tops:
                print(k + ": ", train_net_tops[k])

            # report test result
            __display_test(net, solver, iters)

            # save weight and optimizer state
            __save_snapshot(net, optimizer, iters, solver.snapshot_prefix)


def __display_test(net, solver, iters):
    net.eval()
    print(">>>>>>>>>> start testing...")
    with th.no_grad():
        res = {}
        for i in range(solver.test_iter):
            test_data = solver.get_test_data_batch(solver.test_batchsize, solver.image_size)
            test_net_tops = net.forward(*test_data)
            for k in test_net_tops:
                if k in res.keys():
                    res[k] = res[k] + test_net_tops[k]
                else:
                    res[k] = test_net_tops[k]
        for k in res:
            print(k + ":== ", res[k] / solver.test_iter)

        # log validation loss to file
        with open(solver.test_loss_file, "a") as f:
            f.write(str(iters)+" ")
            f.write(str(res['loss'].cpu().numpy().tolist()/solver.test_iter))
            f.write("\n")
    print(">>>>>>>>>> finish testing ")
    net.train()


def __save_snapshot(net, optimizer, iters, snapshot_prefix):
    print(">>>>>>>>>> saveing modelweight and solverstate...")
    modelweight = net.state_dict()
    solverstate = {
        'optimizer_state_dict': optimizer.state_dict(),
        'iters': iters
    }
    th.save(modelweight, snapshot_prefix +
            "iters_" + str(iters) + ".modelweight")
    th.save(solverstate, snapshot_prefix +
            "iters_" + str(iters) + ".solverstate")
