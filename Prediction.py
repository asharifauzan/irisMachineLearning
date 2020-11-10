import pandas as c 
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 6

irisDataset = c.read_csv('iris.csv', header = 0)
x = irisDataset.iloc[:, :2]
y = irisDataset.iloc[:, -1]

model = KNeighborsClassifier(n_neighbors, weights='distance')
model.fit(x, y)

length = float(input('Masukkan Panjang Sepal (cm) : '))
width = float(input('Masukkan Lebar Sepal (cm) : '))
prediction = model.predict([[length, width]])
print('Prediction : ' + prediction)
