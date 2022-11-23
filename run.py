from Code.generate_data import generate_feature_vectors
from Code.perceptron_training import *
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

if __name__ == '__main__':
    P = int(input("Specify number of feature vectors: "))
    N = int(input("Specify number of components per feature vector: "))
    df = generate_feature_vectors(P, N)
    model = training_perceptron_v2(P,N, 100, df)
    # model = training_perceptron(P,N, 100)
    plot_decision_regions(P, N, clf=model)
    plt.title('Perceptron')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()