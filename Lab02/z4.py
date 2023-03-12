import random
import sklearn
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn import datasets

iris = sklearn.datasets.load_iris()

def iris_test(u_ratio):
    xu = []
    yu = []
    xt = []
    yt = []

    for i in range(0, iris['target'].size):
        if random.random() > u_ratio:
            xu.append(iris['data'][i])
            yu.append(iris['target'][i])
        else:
            xt.append(iris['data'][i])
            yt.append(iris['target'][i])

    neuron = Perceptron(tol = 1e-3, max_iter = 20)
    neuron.fit(xu, yu)
    print("Training/Testing: " + str(round(u_ratio, 2)) + "/" + str(round(1 - u_ratio, 2)))
    print("Accuracy: " + str(round(neuron.score(xt, yt), 3)))

    predicted = neuron.predict(xt)
    confusion_matrix_model = confusion_matrix(yt, predicted)
    print("Confusion matrix:\n" + str(confusion_matrix_model))

iris_test(0.25)
iris_test(0.5)
iris_test(0.75)