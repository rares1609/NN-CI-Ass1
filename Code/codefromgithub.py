import numpy as np

# Basic Rosenblatt Perceptron implementation
class RBPerceptron:

  # Constructor
  def __init__(self, number_of_epochs = 100, learning_rate = 0.1):
    self.number_of_epochs = number_of_epochs
    self.learning_rate = learning_rate

  # Train perceptron
  def train(self, X, D):
    # Initialize weights vector with zeroes
    num_features = X.shape[1]
    self.w = np.zeros(num_features + 1)
    # Perform the epochs
    for i in range(self.number_of_epochs):
      # For every combination of (X_i, D_i)
      for sample, desired_outcome in zip(X, D):
        # Generate prediction and compare with desired outcome
        prediction    = self.predict(sample)
        difference    = (desired_outcome - prediction)
        # Compute weight update via Perceptron Learning Rule
        weight_update = self.learning_rate * difference
        self.w[1:]    += weight_update * sample
        self.w[0]     += weight_update
    return self

  # Generate prediction
  def predict(self, sample):
    outcome = np.dot(sample, self.w[1:]) + self.w[0]
    return np.where(outcome > 0, 1, 0)

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions

# Generate target classes {0, 1}
zeros = np.zeros(50)
ones = zeros + 1
targets = np.concatenate((zeros, ones))

# Generate data
small = np.random.normal(5, 0.25, (50,2))
large = np.random.normal(6.5, 0.25, (50,2))

# Prepare input data
X = np.concatenate((small,large))
D = targets

plt.scatter(small[:,0], small[:,1], color='blue')
plt.scatter(large[:,0], large[:,1], color='red')
plt.show()


rbp = RBPerceptron(600, 0.1)
trained_model = rbp.train(X, D)


plot_decision_regions(X, D.astype(np.integer), clf=trained_model)
plt.title('Perceptron')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()