import torch as th
from torch.autograd import Variable
import numpy as np
import torchvision
import torchvision.transforms as transforms
import math

train_dataset = torchvision.datasets.MNIST(root="~/data/", train=True, transform=transforms.ToTensor, download=True)
test_dataset = torchvision.datasets.MNIST(root="~/data/", train=False, transform=transforms.ToTensor, download=True)

print(train_dataset.train_data.numpy().shape)    # ndarray: [60000, 28, 28]
print(train_dataset.train_labels.numpy().shape)  # ndarray: [60000] :0~9
print(test_dataset.test_data.numpy().shape)      # ndarray: [10000, 28, 28]
print(test_dataset.test_labels.numpy().shape)    # ndarray: [10000] :0~9

# all data: ndarray
X_train = train_dataset.train_data.numpy().reshape([60000, 1, 28, 28])
y_train = train_dataset.train_labels.numpy()
X_test = test_dataset.test_data.numpy().reshape([10000, 1, 28, 28])
y_test = test_dataset.test_labels.numpy()


def get_batch(data, batch, batch_size):
    length = len(data)
    start = batch*batch_size
    end = (batch+1)*batch_size
    if end > length:
        end = length
    return data[start:end]


#===== create model:
class cnn(th.nn.Module):
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


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size()[0], -1)   #!!!
        out = self.fc5(out)
        return out

#=====
network = cnn()

# compatiable for both cpu and gpu
if th.cuda.is_available():
    network = network.cuda()

optimizer = th.optim.Adam(network.parameters(), lr=1e-4)
criterion = th.nn.CrossEntropyLoss()


n_epochs = 10
batch_size = 64

#===== training:
for epoch in range(n_epochs):
    np.random.seed(np.random.randint(0, 99999999))
    shuffled_indices = np.random.permutation(len(X_train))
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    n_batch = math.ceil(X_train.shape[0] / batch_size)   #!!!

    for batch in range(n_batch):
        X_batch = get_batch(X_train, batch, batch_size)
        Y_batch = get_batch(y_train, batch, batch_size)
        
        X_batch_tensor = th.Tensor(X_batch)
        Y_batch_tensor = th.LongTensor(Y_batch)
        #===
        if th.cuda.is_available():  #!!!
            X_batch_tensor, Y_batch_tensor = X_batch_tensor.cuda(), Y_batch_tensor.cuda()
        X_batch_tensor, Y_batch_tensor = Variable(X_batch_tensor), Variable(Y_batch_tensor)
        outputs = network(X_batch_tensor)
        loss = criterion(outputs, Y_batch_tensor)   # criterion(predict, target): predict should be onehot,
                                                    # target should be categarical number.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch : %d Batch : %d Loss : %.3f ' % (epoch, batch, loss.data[0]))
