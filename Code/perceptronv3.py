import numpy as np
import matplotlib.pyplot as plt

def generate_weights(X):
    '''create vector of random weights
    Parameters
    ----------
    X: 2-dimensional array, shape = [n_samples, n_features]
    Returns
    -------
    w: array, shape = [w_bias + n_features]'''
    # rand = np.random.RandomState(random_state)
    # w = rand.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
    w = np.zeros(X.shape[1] + 1)
    print(X.shape[1] + 1)
    return w

def net_input(X, w):
    '''Compute net input as dot product'''
    return np.dot(X, w[1:]) + w[0]

def predict(X, w):
    '''Return class label after unit step'''
    return np.where(net_input(X, w) >= 0.0, 1, -1)

def fit(X, y, eta=0.001, epochs=100):
    '''loop over exemplars and update weights'''
    errors = []
    w = generate_weights(X)

    for _ in range(epochs):
        error = 0

        for xi, target in zip(X, y):

            delta = eta * (target - predict(xi, w))

            w[1:] += delta * xi
            w[0] += delta

            error += int(delta != 0.0)

        errors.append(error)
    return w, errors

def generate_data(P, N, mean1, mean2):
    # create labels vector
    zeros = np.zeros(int(P/2))
    ones = zeros + 1
    minus_ones = zeros - 1
    labels = np.concatenate((minus_ones, ones))
    
    # shuffle labels vector
    np.random.shuffle(labels)

    # create datasets
    small = np.random.normal(mean1, 1, (int(P/2),N))
    large = np.random.normal(mean2, 1, (int(P/2),N))
    df = np.concatenate((small,large))
    
    # show data
    plt.scatter(small[:,0], small[:,1], color='green')
    plt.scatter(large[:,0], large[:,1], color='blue')
    plt.show()

    return df, labels
    
    '''

def data_generator(mu1, sigma1, mu2, sigma2, n_samples, target, seed):
    creates [n_samples, 2] array
    
    Parameters
    ----------
    mu1, sigma1: int, shape = [n_samples, 2]
        mean feature-1, standar-dev feature-1
    mu2, sigma2: int, shape = [n_samples, 2]
        mean feature-2, standar-dev feature-2
    n_samples: int, shape= [n_samples, 1]
        number of sample cases
    target: int, shape = [1]
        target value
    seed: int
        random seed for reproducibility
    
    Return
    ------
    X: ndim-array, shape = [n_samples, 2]
        matrix of feature vectors
    y: 1d-vector, shape = [n_samples, 1]
        target vector
    ------
    X
    rand = np.random.RandomState(seed)
    f1 = rand.normal(mu1, sigma1, n_samples)
    f2 = rand.normal(mu2, sigma2, n_samples)
    X = np.array([f1, f2])
    X = X.transpose()
    y = np.full((n_samples), target)
    return X, y

'''


if __name__ == '__main__':
    dataset, labels = generate_data(100, 2, 0, 0)
    W, errors = fit(dataset, labels, eta=0.01, epochs=200)
    print(len(W))
    y_pred = predict(dataset, labels)
    num_correct_predictions = (y_pred == labels).sum()
    accuracy = (num_correct_predictions / labels.shape[0]) * 100
    print('Perceptron accuracy: %.2f%%' % accuracy)