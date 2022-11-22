from Code.generate_data import generate_feature_vectors

if __name__ == '__main__':
    P = int(input("Specify number of feature vectors: "))
    N = int(input("Specify number of components per feature vector: "))
    generate_feature_vectors(P, N)