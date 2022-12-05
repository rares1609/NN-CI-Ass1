import numpy as np
import matplotlib.pyplot as plt

def generate_data(P, N):
    feature_matrix  = np.random.normal(0, 1, size = (P, N))
    labels = np.random.choice([-1, 1], size = P)
    return feature_matrix, labels

def train_perceptron(N, feature_matrix, labels, learning_rate ,epochs = 100):
    theta = np.zeros((N, N + 1))
    n_miss_list = []
    for epoch in range(epochs):
        no_miss = 0
        for index, x_i in enumerate(feature_matrix):
            x_i = np.insert(x_i, 0, 1).reshape(-1,1)
            y_hat = step_function(np.dot(x_i.T, theta))
            if (np.squeeze(y_hat.any()) - labels[index]) != 0:
                theta += learning_rate*((labels[index] - y_hat)*x_i)
                no_miss += 1
                
        n_miss_list.append(no_miss)
    
    return theta, n_miss_list

def plot_decision_boundary(feature_matrix, theta, labels):
    x1 = [min(feature_matrix[:,0]), max(feature_matrix[:,0])]
    m = -theta[1]/theta[2]
    c = -theta[0]/theta[2]
    x2 = m*x1 + c

    fig = plt.figure(figsize=(10,8))
    plt.plot(feature_matrix[:, 0][labels==1], feature_matrix[:, 1][labels==1], "r^")
    plt.plot(feature_matrix[:, 0][labels==-1], feature_matrix[:, 1][labels==-1], "bs")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title('Perceptron Algorithm')
    plt.plot(x1, x2, 'y-')

def step_function(sample):
    return np.where(sample > 0, 1, -1)


if __name__ == '__main__':
    feature_matrix, labels = generate_data(100, 2)
    print(feature_matrix.shape)
    print(labels.shape)
    theta, miss_l = train_perceptron(5, feature_matrix, labels, 0.5, 100)
    plot_decision_boundary(feature_matrix, theta)

    
    
