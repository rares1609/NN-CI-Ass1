import matplotlib.pyplot as plt
import numpy as np
# from mlxtend.plotting import plot_decision_regions

def generate_data(P, N, mean1, mean2):
    zeros = np.zeros(int(P/2))
    ones = zeros + 1
    minus_ones = zeros - 1
    labels = np.concatenate((minus_ones, ones))
    np.random.shuffle(labels)
    small = np.random.normal(mean1, 1, (int(P/2),N))
    large = np.random.normal(mean2, 1, (int(P/2),N))
    df = np.concatenate((small,large))
    
    plt.scatter(small[:,0], small[:,1], color='green')
    plt.scatter(large[:,0], large[:,1], color='blue')
    plt.show()
    return df, labels



# Generate labels classes {-1, 1}


# Generate data
##small = np.random.normal(0, 1, (50,2))
#large = np.random.normal(0, 1, (50,2))

# Generate data
#small_v1 = np.random.normal(4, 1, (50,2))
#large_v1 = np.random.normal(6.5, 1, (50,2))

# Generate data
#small = np.random.normal(0, 0.1, (50,2))
#large = np.random.normal(0, 0.1, (50,2))

# Prepare input data
#X = np.concatenate((small,large))
#D = labels


