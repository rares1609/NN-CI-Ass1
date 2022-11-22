import numpy as np
import pandas as pd

def generate_feature_vectors(P, N):
    feature_matrix  = np.random.normal(0, 1, size = (P, N))   # Generate P Gaussian distributions with N components each
    labels = np.random.binomial(1, 0.5, size = P)  # Generate P labels 
    dataframe = pd.DataFrame(feature_matrix)  # Convert to dataframe
    dataframe['Labels'] = labels  #  Append labels to dataframe
    print(dataframe)


