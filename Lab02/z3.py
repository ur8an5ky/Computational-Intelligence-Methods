import random
import sklearn
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn import datasets

iris = sklearn.datasets.load_iris()
xu = []
yu = []
xt = []
yt = []

for i in range(0, iris['target'].size):
    if random.random() > 0.2:
        xu.append(iris['data'][i])
        yu.append(iris['target'][i])
    else:
        xt.append(iris['data'][i])
        yt.append(iris['target'][i])

neuron = Perceptron(tol = 1e-3, max_iter = 20)
neuron.fit(xu, yu)
print("Accuracy: " + str(neuron.score(xt, yt)))

predicted = neuron.predict(xt)
confusion_matrix_model = confusion_matrix(yt, predicted)
print("Confusion matrix:\n" + confusion_matrix_model)