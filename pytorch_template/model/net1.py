import torch as th
import torch.nn as nn
import torch.nn.functional as F

class xxx_net(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=1),
            th.nn.Dropout2d(p=0.5),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.conv2 = th.nn.Sequential(
            th.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
            th.nn.Dropout2d(p=0.5),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.conv3 = th.nn.Sequential(
            th.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            th.nn.Dropout2d(p=0.5),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.conv4 = th.nn.Sequential(
            th.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            th.nn.Dropout2d(p=0.5),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.fc5 = th.nn.Linear(in_features=128, out_features=10, bias=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size()[0], -1)
        out = self.fc5(out)
        return out  # logit

    def train_forward(self, x, y):
        assert self.training == True
        logits = self.forward(x)
        criterion = th.nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        # TODO
        # accuracy = 
        ret = {
            "loss" : loss,
            "accuracy" : None
        }
        return ret

    def test_forward(self, x, y):
        assert self.training == False
        # TODO
        ret = {
            "accuracy": None
        }
        return ret

    def deploy_forward(self, x):
        assert self.training == False
        logits = self.forward(x)
        prob = F.softmax(logits)
        return prob
