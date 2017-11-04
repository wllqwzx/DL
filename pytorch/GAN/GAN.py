import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transform

trans = transform.Compose([transform.ToTensor(),
                           transform.Normalize(mean=(0,0,0), std=(1,1,1))])


train_dsets = dsets.MNIST("../data/",
                          train=True,
                          transform=trans,
                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dsets,
                                           batch_size=100,
                                           shuffle=True)


class Discrimininator(nn.Module):
    def __init__(self):
        super(Discrimininator,self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,1)

    def forward(self, input):
        out = nn.functional.relu(self.fc1(input))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.sigmoid(self.fc3(out))
        return out

        
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,784)

    def forward(self, input):
        out = nn.functional.leaky_relu(self.fc1(input))
        out = nn.functional.leaky_relu(self.fc2(out))
        out = nn.functional.tanh(self.fc3(out))
        return out


discrime = Discrimininator()
gener = Generator()

loss = nn.BCELoss()
d_optimizer = torch.optim.Adam(params = discrime.parameters(), lr=0.0005)
g_optimizer = torch.optim.Adam(params = gener.parameters(), lr = 0.0005)



# training
for epoch in range(200):
    for i, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1)
        images = Variable(images)
        real_flag = Variable(torch.ones(images.size(0)))
        fake_flag = Variable(torch.zeros(images.size(0)))

        discrime.zero_grad()
        real_out = discrime(images)
        real_loss = loss(real_out, real_flag)

        noise = Variable(torch.randn(images.size(0), 64)) 
        fake_images = gener(noise)
        fake_out = discrime(fake_images)
        fake_loss = loss(fake_out, fake_flag)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        gener.zero_grad()
        noise = Variable(torch.randn(images.size(0), 64))
        fake_images = gener(noise)
        fake_out = discrime(fake_images)
        g_loss = loss(fake_out, real_flag)
        g_loss.backward()
        g_optimizer.step()

        if (i % 300) == 0:
            print("epoch[%d/%d], step[%d/%d], d_loss:%.4f, g_loss:%.4f, D(x):%.2f, D(G(z)):%.2f" %
                  (epoch, 200, i, 600, d_loss.data[0], g_loss.data[0], real_out.data.mean(), fake_out.data.mean()))

    # generate a image in each epoch
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(fake_images.data, './data/fake_samples_%d.png' %(epoch+1))


