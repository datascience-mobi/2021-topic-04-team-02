import matplotlib.pyplot as plt
import numpy
import pandas as pd

#laden der .csv in ein Array
images = numpy.loadtxt('/Users/jojo3/PycharmProjects/pythonstuff/data/mnist_train.csv',  delimiter=',')
#image = image[:,1:] #erste Spalte ist die "Legende" für die im Bild codierte Zahl
#X = image[6,:].reshape(28,28) #X = Bild in der 6. zeile der.csv
print(images.shape)

dataset = pd.DataFrame(images)
dataset.head()



