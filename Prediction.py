import seaborn as a 
import matplotlib.pyplot as b 
import pandas as c 
import numpy as d
from sklearn import neighbors, datasets
from sklearn import preprocessing 

n_neighbors = 6

irisDataset = c.read_csv('iris.csv', header = 0)
x = irisDataset.iloc[:, :2]
y = irisDataset.iloc[:, -1]
h = .02

model = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
model.fit(x, y)

length = input('Masukkan Panjang Sepal (cm) : ')
width = input('Masukkan Lebar Sepal (cm) : ')
prediction = model.predict([[length, width]])
print('Prediction : ')

if prediction == 0:
    print('Iris Sentosa')
elif prediction == 1:
    print("Iris Versicolor")
else:
    print("Iris Virginica")