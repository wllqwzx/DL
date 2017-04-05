import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import visualize

train_dataset = dsets.MNIST(root="../data/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root="../data/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)




# network
#===================================
# naive version
class inception(nn.Module):
    def __init__(self, ):
        super(inception,self).__init__()


    def forward(self,x):


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1,6,5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Conv2d(6,16,5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(nn.Linear(16*4*4,120),
                                    nn.ReLU(),
                                    nn.Linear(120,84),
                                    nn.ReLU(),
                                    nn.Linear(84,10))

    def forward(self,images):
        images = images.view(images.size(0),1,28,28)
        out = self.layer1(images)
        out = out.view(out.size(0),16*4*4)
        out = self.layer2(out)
        return out


net = GoogLeNet()
#===================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

has_output = False
for epoch in range(5):
    for i,(images,labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        net.zero_grad()
        output = net(images)
        if has_output == False:
            visualize.make_dot(output).render("graph")
            has_output = True
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()

        if(i%200 == 0):
            print("epoch:%d, batch:%d, loss:%.4f" % (epoch,i,loss.data[0]))


total = 0.0
correct = 0.0
for images,labels in test_loader:
    images = Variable(images)
    output = net(images)
    val, index = torch.max(output,1)
    total += images.size(0)
    correct += (index.data == labels).sum()

print("Accuract of %d test set is %.2f" % (total, 100*correct/total))
