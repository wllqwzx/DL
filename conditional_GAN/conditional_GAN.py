import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torchvision import datasets
from torchvision import transforms


trans = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1,1,1))])

train_dsets = datasets.MNIST("~/data/",
                          train=True,
                          transform=trans,
                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dsets,
                                           batch_size=100,
                                           shuffle=True)


class Discriminate(nn.Module):
    def __init__(self):
        super(Discriminate,self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features = 512)
        self.bn1 = nn.BatchNorm1d(num_features = 512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, input):
        out = nn.functional.relu(self.bn1(self.fc1(input)))
        out = nn.functional.relu(self.bn2(self.fc2(out)))
        out = nn.functional.sigmoid(self.fc3(out))
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(in_features=32, out_features = 256)    # 32: noise
        self.fcx = nn.Linear(in_features=10, out_features=256)      # 10: label
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.bn2 = nn.BatchNorm1d(num_features = 512)
        self.fc3 = nn.Linear(in_features=512, out_features=784)

    def forward(self, input, labels):
        onehot_labels = torch.zeros(100,10)
        for i in range(100):
            onehot_labels[i][labels.data[i]] = 1
        #labels = Variable(labels.data.view(100,1))
        #labels = labels.float()
        onehot_labels = Variable(onehot_labels)
        out = self.fc1(input)
        outx = self.fcx(onehot_labels)
        #out = nn.functional.leaky_relu(self.bn1(out))
        out = nn.functional.leaky_relu(self.bn1(out+outx))
        out = nn.functional.leaky_relu(self.bn2(self.fc2(out)))
        out = nn.functional.sigmoid(self.fc3(out))
        return out


discri = Discriminate()
genera = Generator()

loss = nn.CrossEntropyLoss()
d_optimizer = optim.Adam(params = discri.parameters(), lr=0.0001)
g_optimizer = optim.Adam(params = genera.parameters(), lr=0.0001)


for epoch in range(200):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(images.size(0), -1))
        labels = Variable(labels)

        real_labels = discri(images)
        real_loss = loss(real_labels, labels)

        noise = Variable(torch.randn(images.size(0), 32))
        fake_images = genera(noise, labels)
        fake_labels = discri(fake_images)
        fake_loss = loss(fake_labels, labels)

        d_loss = real_loss + fake_loss
        discri.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        noise = Variable(torch.randn(100, 32))
        made_labels = torch.rand(100)
        c = 0
        for k in range(10):
            for j in range(10):
                made_labels[c] = k
                c = c + 1
        made_labels = Variable(made_labels).long()
        fake_images = genera(noise, made_labels)
        fake_labels = discri(fake_images)
        fake_loss = loss(fake_labels, made_labels)
        genera.zero_grad()
        fake_loss.backward()
        g_optimizer.step()

        if (i%100) == 0 :
            print("epoch[%d/%d], batch[%d/%d], D(x):%.2f , D(G(z)):%.2f"  
                   % (epoch,200, i, 600, real_loss.data[0], fake_loss.data[0]))

    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(fake_images.data, './data/fake_samples_%d.png' %(epoch+1))
    