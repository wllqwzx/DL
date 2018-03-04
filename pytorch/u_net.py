import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

def make_convlayer(in_c, out_c, drop_r):
    return th.nn.Sequential(
        th.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3,3), stride=1, padding=1),
        # th.nn.Dropout2d(p=drop_r),
        th.nn.BatchNorm2d(out_c),
        th.nn.ReLU(),
        th.nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3,3), stride=1, padding=1),
        th.nn.BatchNorm2d(out_c),
        th.nn.ReLU()
    )


class U_Net(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = make_convlayer(3, 32, 0.1)
        self.p1 = th.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.c2 = make_convlayer(32, 64, 0.1)
        self.p2 = th.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.c3 = make_convlayer(64, 128, 0.2)
        self.p3 = th.nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.c4 = make_convlayer(128, 256, 0.2)
        self.p4 = th.nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.c5 = make_convlayer(256, 512, 0.3)

        self.u6 = th.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=2, padding=0)
        self.c6 = make_convlayer(512, 256, 0.2)

        self.u7 = th.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2,2), stride=2, padding=0)
        self.c7 = make_convlayer(256, 128, 0.2)

        self.u8 = th.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2, padding=0)
        self.c8 = make_convlayer(128, 64, 0.2)

        self.u9 = th.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2,2), stride=2, padding=0)
        self.c9 = make_convlayer(64, 32, 0.2)

        self.logits = th.nn.Conv2d(32, 1, kernel_size=(1,1), stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        oc1 = self.c1(x)
        op1 = self.p1(oc1)
        
        oc2 = self.c2(op1)
        op2 = self.p2(oc2)
        
        oc3 = self.c3(op2)
        op3 = self.p3(oc3)

        oc4 = self.c4(op3)
        op4 = self.p4(oc4)
        
        oc5 = self.c5(op4)

        ou6 = self.u6(oc5)
        ou6 = th.cat([oc4, ou6], dim=1)
        oc6 = self.c6(ou6)

        ou7 = self.u7(oc6)
        ou7 = th.cat([oc3, ou7], dim=1)
        oc7 = self.c7(ou7)

        ou8 = self.u8(oc7)
        ou8 = th.cat([oc2, ou8], dim=1)
        oc8 = self.c8(ou8)

        ou9 = self.u9(oc8)
        ou9 = th.cat([oc1, ou9], dim=1)
        oc9 = self.c9(ou9)

        out = self.logits(oc9)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

