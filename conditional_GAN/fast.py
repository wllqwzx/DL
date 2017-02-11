import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transform

trans = transform.Compose([transform.ToTensor(),
                           transform.Normalize(mean=(0.5,0.5,0.5), std=(1,1,1))])

train_dsets = dsets.MNIST("../data/",
                          train=True,
                          transform=trans,
                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dsets,
                                           batch_size=100,
                                           shuffle=True)

#D
class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.layers=nn.Sequential(nn.Linear(784,512),
                                  nn.ReLU(),
                                  nn.Linear(512,128),
                                  nn.ReLU(),
                                  nn.Linear(128,10))
    
    def forward(self,input):
        return nn.functional.sigmoid(self.layers(input))

#G
class G(nn.Module):
    def __init__(self):
        super(G,self).__init__()
        self.layers = nn.Sequential(nn.Linear(11,256),
                      nn.LeakyReLU(),
                      nn.Linear(256,512),
                      nn.LeakyReLU(),
                      nn.Linear(512,784))

    def forward(self,noise,label):
        mlabels = torch.zeros(label.size(0),10)
        for i in range(label.size(0)):
            mlabels[i][label.data[i]] = 1
        mlabels = Variable(mlabels)
        inpt = torch.cat((noise,mlabels),1)
        return nn.functional.tanh(self.layers(inpt))

dis = D()
gen = G()
loss = nn.CrossEntropyLoss()
dis_optim = torch.optim.Adam(dis.parameters(), lr=1e-3)
gen_optim = torch.optim.Adam(gen.parameters(), lr=1e-3)
fake_images = 0

for epoch in range(200):
    for i,(images,labels) in enumerate(train_loader):
        images = Variable(images.view(images.size(0),-1))
        labels = Variable(labels)
        dis.zero_grad()
        real_labels = dis(images)
        real_loss = loss(real_labels,labels)
        noise = Variable(torch.randn(images.size(0),1))
        fake_images = gen(noise,labels)
        fake_labels = dis(fake_images)
        fake_loss = fake_labels.data.mean()
        two_loss = real_loss + fake_loss
        two_loss.backward()
        dis_optim.step()

        gen.zero_grad()
        noise = Variable(torch.randn(images.size(0),1))
        fake_images = gen(noise,labels)
        fake_labels = dis(fake_images)
        fake_loss = loss(fake_labels, labels)
        fake_loss.backward()
        gen_optim.step()

        if(i%200 == 0):
            print("epoch:%d, i:%d, D(x):%.2f, G(D(z)):%.2f" %
                   (epoch,i,real_loss.data[0],fake_loss.data[0]))
    
    torchvision.utils.save_image(fake_images.view(fake_images.size(0),1,28,28).data,"fake_image%d.png" % epoch)

