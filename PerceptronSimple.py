import numpy as np
import os
import matplotlib.pyplot as plt

def generate_data(n):
    """
    This function generates a linearly separable dataset with 2 features.

    Args:
    n: size of the dataset

    Returns:
    A tuple containing two lists:
        - features: list of n-dimensional vectors representing features
        - labels: list of binary labels (0 or 1) representing classes
    """

    # Generate random linearly separable data
    mean1 = np.random.randint(low=-10, high=0, size=(2,))
    mean2 = np.random.randint(low=0, high=10, size=(2,))
    cov = np.identity(2) * 4
    x1 = np.random.multivariate_normal(mean1, cov, n // 2)
    x2 = np.random.multivariate_normal(mean2, cov, n // 2)

    # Concatenate the data and labels
    features = np.concatenate([x1, x2], axis=0)
    labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

    # Shuffle the data
    idx = np.random.permutation(n)
    features = features[idx]
    labels = labels[idx]

    return features.tolist(), labels.tolist()


def plot_data(features, labels,filename=None, weight_vector=None,format='png'):
    """
    This function plots the generated data with the labels and saves it as an image.

    Args:
    features: list of n-dimensional vectors representing features
    labels: list of binary labels (0 or 1) representing classes
    filename: name of the file to save the image
    format: format of the image (default: png)

    Returns:
    None
    """

    plt.clf()

    # Convert lists to arrays
    features = np.array(features)
    labels = np.array(labels)

    # Plot the data
    plt.scatter(features[:, 0], features[:, 1], c=labels)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linearly Separable Dataset')

    if weight_vector is not None:
        # Define the x and y coordinates for the decision boundary
        x = np.linspace(np.min(features[:, 0]), np.max(features[:, 0]), 100)
        y = (-weight_vector[0] * x) / weight_vector[1]

        # Plot the decision boundary
        plt.plot(x, y, color='green', label='Decision boundary')

    # Save the plot as an image
    if filename is not None:
        plt.savefig(filename, format=format)

    plt.show()


def step(x):
    if x > 0:
        return 1
    return 0


class Perceptron:
    def __init__(self, input_size, learning_rate = 0.01) -> None:
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_size,1)
        self.bias = np.random.randn()

    def classify(self,X):
        z = np.dot(X,self.weights) + self.bias
        return step(z)
 
    def train(self, training_features, training_labels, epochs = 100):
        for epoch in range(epochs):
            total_error = 0
            for X, label in zip(training_features,training_labels):
                error = label - self.classify(X)
                #print(f"{error=}")

                self.bias += self.learning_rate * error
                for i in range(len(self.weights)):
                    #print(f"{self.weights[i]=}{self.learning_rate=}{error=}{X[i]=}")
                    self.weights[i] += self.learning_rate * error * X[i]

                total_error += abs(error)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Error = {total_error}")

            elif total_error == 0:
                print(f"Epoch {epoch}: Error = {total_error}")
                break
        print(f"Epoch {epoch}: Error = {total_error}")
        
    
    def eval(self ,features,labels):
        correct = 0
        for x,y in zip(features,labels):
            if self.classify(x) == y:
                correct += 1
        
        return f"{(correct/len(features))*100}%"
    
    def out(self):
        print(f"weights\n{self.weights}\n\nbias\n{self.bias}")

np.random.seed(890)

p1 = Perceptron(2)

training_features, training_labels = generate_data(1000)

plot_data(training_features,training_labels,"pretrain.png",p1.weights)


print(p1.eval(training_features,training_labels))
p1.train(training_features,training_labels,1000)
print(p1.eval(training_features,training_labels))
plot_data(training_features,training_labels,"postrain.png",p1.weights)

