import matplotlib.pyplot as plt
import numpy as np
# from mlxtend.plotting import plot_decision_regions

# Generate labels classes {-1, 1}
zeros = np.zeros(50)
ones = zeros + 1
minus_ones = zeros - 1
labels = np.concatenate((minus_ones, ones))
np.random.shuffle(labels)

# Generate data
small = np.random.normal(0, 1, (50,2))
large = np.random.normal(0, 1, (50,2))

# Generate data
small_v1 = np.random.normal(4, 1, (50,2))
large_v1 = np.random.normal(6.5, 1, (50,2))

# Generate data
small = np.random.normal(0, 0.1, (50,2))
large = np.random.normal(0, 0.1, (50,2))

# Prepare input data
X = np.concatenate((small,large))
D = labels

plt.scatter(small_v1[:,0], small_v1[:,1], color='green')
plt.scatter(large_v1[:,0], large_v1[:,1], color='blue')
plt.show()

plt.scatter(small[:,0], small[:,1], color='green')
plt.scatter(large[:,0], large[:,1], color='blue')
plt.show()

print(labels)