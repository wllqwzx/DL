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

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=10,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)




# network
#===================================
class inception(nn.Module):
    def __init__(self, in_chn, b1_chn, b2_chn1, b2_chn2, b3_chn1, b3_chn2, b4_chn):
        super(inception,self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_chn,b1_chn,1),
                                nn.BatchNorm2d(b1_chn),
                                nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_chn,b2_chn1,1),
                                nn.BatchNorm2d(b2_chn1),
                                nn.ReLU(True),
                                nn.Conv2d(b2_chn1,b2_chn2,3,1,1),
                                nn.BatchNorm2d(b2_chn2),
                                nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_chn,b3_chn1,1),
                                nn.BatchNorm2d(b3_chn1),
                                nn.ReLU(True),
                                nn.Conv2d(b3_chn1,b3_chn2,5,1,2),
                                nn.BatchNorm2d(b3_chn2),
                                nn.ReLU(True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3,1,1),
                                nn.Conv2d(in_chn, b4_chn,1),
                                nn.BatchNorm2d(b4_chn),
                                nn.ReLU(True))

    def forward(self,x):
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)
        out4 = self.b4(x)
        return torch.cat((out1,out2,out3,out4),1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        self.prelayers = nn.Sequential(nn.Conv2d(1,192,3,1,1),
                                       nn.BatchNorm2d(192),
                                       nn.ReLU(True))
        self.a3 = inception(192,64,96,128,16,32,32)
        self.b3 = inception(256,128,128,192,32,96,64)
        self.pool = nn.MaxPool2d(2,2)
        self.a4 = inception(480,192,96,208,16,48,64)
        self.b4 = inception(512,160,112,224,24,64,64)
        self.c4 = inception(512,128,128,256,24,64,64)
        self.d4 = inception(512,112,144,288,32,64,64)
        self.e4 = inception(528,256,120,320,32,128,128)
        #pool
        self.a5 = inception(832,256,160,320,32,128,128)
        self.b5 = inception(832,384,192,384,48,128,128)
        self.avgpool = nn.AvgPool2d(7)
        self.drop = nn.Dropout2d(0.4)
        self.linear = nn.Linear(1024,10)


    def forward(self,images):
        images = images.view(images.size(0),1,28,28)
        out = self.prelayers(images)
        out = self.a3(out)
        out = self.b3(out)
        out = self.pool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.pool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = self.drop(out)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        out = nn.functional.softmax(out)
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
