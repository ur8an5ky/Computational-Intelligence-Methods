from sklearn.linear_model import Perceptron
import numpy as np
from matplotlib import pyplot as plt

matrix = np.loadtxt("/content/drive/MyDrive/Colab/fuel.txt", dtype=float)
properties = []
cleanliness = []
# print(matrix)
for i in range(len(matrix)):
  properties.append(matrix[i][:3])
  cleanliness.append(matrix[i][3])

neuron = Perceptron(tol = 1e-3, max_iter = 5)
neuron.fit(properties, cleanliness)
print(neuron.score(properties, cleanliness))