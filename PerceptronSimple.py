import numpy as np

def generate_data(n):
    features = []
    labels = []

    for _ in range(n):
        temp = np.random.randint(0, 41)

        humidity = np.random.randint(0, 101)

        feature_vector = np.array([temp, humidity])
        features.append(feature_vector)

        # Label the data point as 1 (good) if temperature is above 25 degrees Celsius and humidity is below 50%, otherwise 0 (bad)
        if temp > 25 and humidity < 50:
            labels.append(1)
        else:
            labels.append(0)

    return features, labels


def step(x):
    if x > 0:
        return 1
    return 0


class Perceptron:
    def __init__(self, input_size, learning_rate = 0.1) -> None:
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_size,1)
        self.bias = np.random.randn()

    def classify(self,X):
        z = np.dot(X,self.weights) + self.bias
        return step(z)
 
    def train(self, features, labels, epochs = 100):
        for epoch in range(epochs):
            total_error = 0
            for X, label in zip(features,labels):
                error = label - self.classify(X)
                #print(f"{error=}")

                self.bias += self.learning_rate * error
                for i in range(len(self.weights)):
                    #print(f"{self.weights[i]=}{self.learning_rate=}{error=}{X[i]=}")
                    self.weights[i] += self.learning_rate * error * X[i]

                total_error += abs(error)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Error = {total_error}")    
            
    
    def out(self):
        print(f"weights\n{self.weights}\n\nbias\n{self.bias}")

np.random.seed(22)

training_features, training_labels = generate_data(10000)


p1 = Perceptron(2)
p1.train(training_features,training_labels,1000)

#print(training_features[3])

index = 9
print(p1.classify(training_features[index]),training_labels[index])