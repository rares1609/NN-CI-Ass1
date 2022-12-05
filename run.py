from Code.generate_data_v1 import generate_data
from Code.perceptron_training import *
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

if __name__ == '__main__':
    P = int(input("Specify number of feature vectors: "))
    N = int(input("Specify number of components per feature vector: "))
    df, labels = generate_data(P, N, 6.5, 4)
    print(df)
    print(labels)
