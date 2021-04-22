
#import pandas as pd
#laden der .csv in ein Array
#images = numpy.loadtxt('/data/mnist_train.csv',  delimiter=',')
#image = image[:,1:] #erste Spalte ist die "Legende" für die im Bild codierte Zahl
#X = image[6,:].reshape(28,28) #X = Bild in der 6. zeile der.csv
#print(images.shape)
#dataset = pd.DataFrame(images)
#dataset.head()
import numpy as np
import pickle
X = np.genfromtxt('data/mnist_train.csv', delimiter=',',skip_header=1).astype(np.dtype('uint8'))
with open('data/train_points.p', 'wb') as f:
    pickle.dump(X, f)
...
with open('data/train_points.p', 'rb') as f:
    X = pickle.load(f)

