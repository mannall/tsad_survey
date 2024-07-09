import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 5000

x = np.linspace(0, 1, num=N).reshape(-1, 1)

train_y = np.sin(2*4*np.pi*x) + 0.2*np.random.random(x.shape)
np.savetxt("example.train.csv", train_y, delimiter=',', fmt='%.4f')

test_y = np.sin(2*4*np.pi*x) + 0.2*np.random.random(x.shape)
test_y[2500] += 1

labels = np.zeros((x.size, 1))
labels[2500] = 1

np.savetxt("example.test.csv", np.hstack((test_y, labels)), delimiter=',', fmt='%.4f')

plt.plot(train_y)
plt.plot(test_y)
plt.show()