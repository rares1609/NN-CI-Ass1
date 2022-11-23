import numpy as np

def training_perceptron(N, P, no_epochs):
    learning_rate = 0.1
    weights = np.zeros(P + 1)

    for i in range(no_epochs):
        for sample, optimal_outcome in zip(P, N):
            prediction = predict(sample)
            diff = optimal_outcome - prediction
            updated_weight = learning_rate * diff
            weights[1:] = weights[1:] + updated_weight * sample
            weights[0] = weights[0] + updated_weight


def predict(sample, w):
    result = np.dot(sample, w[1:]) + w[0]
    return np.where(result > 0, 1, 0)

def training_perceptron_v2(N, P, no_epochs, dataframe):
    learning_rate = 0.1
    vector = np.vectorize(np.int_)
    weights = np.zeros(P + 1)
    weights = vector(weights)
    #print(weights)
    
    for iteration in range(no_epochs):
        for i in range (P):
            for j in range (N):
                prediction = predict(dataframe[i][j], weights)
                diff = dataframe[i][j] - prediction
                updated_weight = learning_rate * diff
                weights[1:] = weights[1:] + updated_weight * dataframe[i][j]
                print(weights)
                weights[0] = weights[0] + updated_weight
    return weights