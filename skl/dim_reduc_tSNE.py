from sklearn.manifold import TSNE
from time import time
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/")
n = 1000
x_data = mnist.train.images[:n]
y_data = mnist.train.labels.astype("int")[:n]


#=== perform t-SNE embedding
t0 = time()
tsne = TSNE(n_components=2, init="random")
vis_data = tsne.fit_transform(x_data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]
plt.scatter(vis_x, vis_y, c=y_data, s=5, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)

plt.title("t-SNE for mnist, time: %.2f s" % (time()-t0) )
plt.xlabel("dim1")
plt.ylabel("dim2")
plt.show()

