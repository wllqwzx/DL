import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

num_epochs = 10
batch_size = 100
learning_rate = 0.001

class MinstDataSet(data.Dataset):
	def __init__(self):
		trainX = pd.read_csv("train.csv")
		trainY = trainX.label.as_matrix().tolist()
		trainX = trainX.drop("label",axis=1).as_matrix().reshape(42000,1,28,28)
		self.datalist = trainX
		self.labellist = trainY


	def __getitem__(self, index):
		return torch.Tensor(self.datalist[index].astype(float)), self.labellist[index]

	def __len__(self):
		return self.datalist.shape[0]


train_dataset = MinstDataSet()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True,
                                           num_workers=2)


class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=5),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.fc = nn.Linear(4*4*32, 10)


	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0),-1)	# the size -1 means it is inferred from other dimensions 
		out = self.fc(out)
		return out



cnn = CNN()



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)


for epoch in range(10):
	for i, (images, labels) in enumerate(train_loader):
		images = Variable(images)
		labels = Variable(labels)

		optimizer.zero_grad()
		outputs = cnn(images)

		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, 10, i+1, len(train_dataset)//batch_size, loss.data[0]))


cnn.eval()
testX = pd.read_csv("test.csv")
testX = testX.as_matrix().reshape(28000,1,28,28).astype(float)
testX = Variable(torch.Tensor(testX))
pred = cnn(testX)
_, predlabel = torch.max(pred.data, 1)
predlabel = predlabel.tolist()

predlabel = pd.DataFrame(predlabel)
predlabel.index = np.arange(28000) + 1
id = np.arange(28000) + 1
id = pd.DataFrame(id)
id.index = id.index + 1

predlabel = pd.concat([id,predlabel], axis=1)
predlabel.columns = ["ImageId", "Label"]

predlabel.to_csv('predict.csv', index= False)
