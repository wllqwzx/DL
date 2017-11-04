import numpy as np
import torch as th
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms


# config:
lr=0.0002
image_size = 64 #iamge = 3 * 64 * 64
nz=100      # 噪声维度(channel)
nc=3        # 图片三通道
bh_size=32  
workers=2   


trans = transforms.Compose([transforms.Scale(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

train_dataset = datasets.CIFAR10(root="/data/",
                                 train=True,
                                 transform=trans,
                                 download=True)

train_loader = th.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=bh_size,
                                        shuffle=False,
                                        num_workers=workers)



class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(name="deconv1",module=nn.ConvTranspose2d(in_channels=nz, out_channels=512, kernel_size=4, stride=1, padding=0,bias=False))
        self.layers.add_module(name="bn1", module=nn.BatchNorm2d(num_features=512))
        self.layers.add_module(name="relu1", module=nn.ReLU(True))
        self.layers.add_module(name="deconv2",module=nn.ConvTranspose2d(in_channels=512, out_channels=256,kernel_size=4, stride=2, padding=1,bias=False))
        self.layers.add_module(name="bn2", module=nn.BatchNorm2d(num_features=256))
        self.layers.add_module(name="relu2", module=nn.ReLU(inplace=True))
        self.layers.add_module(name="deconv3", module=nn.ConvTranspose2d(in_channels=256, out_channels=128,kernel_size=4, stride=2, padding=1,bias=False))
        self.layers.add_module(name="bn3", module=nn.BatchNorm2d(num_features=128))
        self.layers.add_module(name="relu3", module=nn.ReLU(True))
        self.layers.add_module(name="deconv4", module=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4, stride=2, padding=1,bias=False))
        self.layers.add_module(name="bn4", module=nn.BatchNorm2d(64))
        self.layers.add_module(name="relu4", module=nn.ReLU(True))
        self.layers.add_module(name="deconv5", module=nn.ConvTranspose2d(in_channels=64, out_channels=3,kernel_size=4, stride=2, padding=1,bias=False))
        self.layers.add_module(name="tanh", module=nn.Tanh())

    def forward(self,input):
        return self.layers(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('conv1',nn.Conv2d(nc,64,4,2,1,bias=False))
        self.layers.add_module('relu1',nn.LeakyReLU(0.2,inplace=True))
        
        self.layers.add_module('conv2',nn.Conv2d(64,128,4,2,1,bias=False))
        self.layers.add_module('bnorm2',nn.BatchNorm2d(128))
        self.layers.add_module('relu2',nn.LeakyReLU(0.2,inplace=True))
        
        self.layers.add_module('conv3',nn.Conv2d(128,256,4,2,1,bias=False))
        self.layers.add_module('bnorm3',nn.BatchNorm2d(256))
        self.layers.add_module('relu3',nn.LeakyReLU(0.2,inplace=True))
        
        self.layers.add_module('conv4',nn.Conv2d(256,512,4,2,1,bias=False))
        self.layers.add_module('bnorm4',nn.BatchNorm2d(512))
        self.layers.add_module('relu4',nn.LeakyReLU(0.2,inplace=True))
        
        self.layers.add_module('conv5',nn.Conv2d(512,1,4,1,0,bias=False))
        self.layers.add_module('sigmoid',nn.Sigmoid())

    def forward(self,input):
        return self.layers(input)


g_net = Generator()
d_net = Discriminator()

g_optimizer = th.optim.Adam(params=g_net.parameters(), lr=lr)
d_optimizer = th.optim.Adam(params=d_net.parameters(), lr=lr)
loss = nn.BCELoss()


for epoch in range(100):
    for i,(real_images,_) in enumerate(train_loader):

        # optimize d_net
        real_labels = Variable(th.ones(bh_size))
        fake_labels = Variable(th.zeros(bh_size))
        d_net.zero_grad()
        real_images = Variable(real_images)
        real_out = d_net(real_images)
        real_loss = loss(real_out, real_labels)

        noise = Variable(th.randn(real_images.size(0),100,1,1))
        fake_images = g_net(noise)
        fake_out = d_net(fake_images)
        fake_loss = loss(fake_out, fake_labels)

        two_loss = fake_loss + real_loss
        two_loss.backward()
        d_optimizer.step()

        # optimize g_net (optimize times: g_net:d_net = 1:2)
        if th.rand(1)[0] > 0.5:
            g_net.zero_grad()
            noise = Variable(th.randn(real_images.size(0),100,1,1))
            fake_images = g_net(noise)
            fake_out = d_net(fake_images)
            fake_loss = loss(fake_out, real_labels)
            fake_loss.backward()
            g_optimizer.step()
        
        if i%10 == 0:
            print("epoch:%d, batch:%d, D(x):%d, D(G(z)):%d" %
                   (epoch, i*32, real_out.data.mean(), fake_out.data.mean()))
            fake_images = fake_images.view(fake_images.size(0), 3, 64, 64)
            torchvision.utils.save_image(fake_images.data, 'fake_samples_%d.png' % i)
