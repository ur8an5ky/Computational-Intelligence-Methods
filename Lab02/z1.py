import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

xt1 = np.random.normal([0,1],1,size=(200,2))
yt1 = np.zeros(200,dtype='int')
xt2 = np.random.normal([1,1],1,size=(200,2))
yt2 = np.ones(200,dtype='int')

xt = np.concatenate((xt1,xt2))
yt = np.concatenate((yt1,yt2))

accuracy = {}

for i, val in enumerate([5,10,20,100]):
    fig,ax = plt.subplots()
    xu1 = np.random.normal([0,1],1,size=(val,2))
    yu1 = np.zeros(val,dtype='int')
    xu2 = np.random.normal([1,1],1,size=(val,2))
    yu2 = np.ones(val,dtype='int')

    xu = np.concatenate((xu1 ,xu2))
    yu = np.concatenate((yu1, yu2))

    neuron = Perceptron(tol=1e-4, max_iter = 100)

    neuron.fit(xu,yu)

    accuracy[val] = neuron.score(xt,yt)

    x1 = np.linspace(-4,4,100)
    x2 = -(1./neuron.coef_[0][1])*(neuron.coef_[0][0]*x1+neuron.intercept_[0])
    ax.plot(x1, x2, '-r')

    ax.scatter(np.array(xu1)[:,0], np.array(xu1)[:,1])
    ax.scatter(np.array(xu2)[:,0], np.array(xu2)[:,1])

    fig.show()