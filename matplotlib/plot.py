import matplotlib.pyplot as plt
from torchnet.meter import MovingAverageValueMeter

train_file = "train_loss.txt"
text_file = "test_loss.txt"
save_file = "loss_curve.pdf"

train_f = open(train_file)
train_d = train_f.readlines()
train_f.close()

valid_f = open(text_file)
valid_d = valid_f.readlines()
valid_f.close()

train_iter = []
train_loss = []
i = 0
ma_loss = MovingAverageValueMeter(windowsize=500)
for s in train_d:
    i = i+1
    t = s.strip().split(' ')
    t_iter = int(t[0])
    ma_loss.add(float(t[1]))
    if i % 500 == 0:
        train_iter.append(t_iter)
        train_loss.append(ma_loss.value()[0])

valid_iter = []
valid_loss = []
i = 0
for s in valid_d:
    i = i + 1
    if i>=0:
        t = s.strip().split(' ')
        t_iter = int(t[0])
        t_loss = float(t[1])
        valid_iter.append(t_iter)
        valid_loss.append(t_loss)

#==========
#plt.semilogx(x, b, marker='^', linewidth=0.5, color='k')
#plt.semilogx(x, a, marker='o', linewidth=0.5, color='k')

# color: b: , o: , g: green
plt.plot(train_iter, train_loss, linewidth=1, linestyle='-.', color='blue')
plt.plot(valid_iter, valid_loss, linewidth=1, color='red')
plt.legend(["ResNet50 train (avg500batch)","ResNet50 val"], loc="upper right")
plt.xlabel("iters")
plt.ylabel("loss")
plt.grid(linestyle='dotted')

fig = plt.gcf()
# fig.set_size_inches(6.5, 4)
fig.savefig(save_file, dpi=100)
# plt.show()