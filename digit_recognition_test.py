import matplotlib.pyplot as plt
import numpy

image = numpy.loadtxt('/Users/jojo3/PycharmProjects/pythonstuff/data/mnist_train.csv',  delimiter=',')
image = image[:,1:]
X = image[6,:].reshape(28,28)
print(image.shape)
figure()
plt.imshow(X)



