import numpy as np
import pandas as pd

class Perceptron():
    def __init__(self):

        self.weights = None
        self.learning_rate = 0.02
        self.num_epochs = 1000

    def feature_extraction(self, data):
        attributes = data.columns[:-1]
        return data[attributes], data['decision']

    def sgn_function(self, perceptron_input):
        # Sign function for sigmoid input
        if perceptron_input>0.5:
            return 1.0
        return -1.0

    def update_weights(self, new_weights):
        # Modularity for weight update
        self.weights = new_weights

    def train(self, labeled_data, learning_rate=None, max_iter=None):

        if max_iter != None:
            self.num_iterations = max_iter
        
        if learning_rate != None:
            self.learning_rate = learning_rate

        features, labels = self.feature_extraction(labeled_data)
        features = np.array(features, dtype='float64')
        labels = np.array(labels, dtype='float64')
        labels = np.where(labels<1,-1,labels)

        # Need to add the bias term as well! one more term
        bias = [[1] for x in range(len(features))]
        features = np.append(features, bias, axis=1)
        self.weights = np.zeros((features.shape[1]))
        self.sample_size = len(features)
        # Using the margin concept to prevent overfitting. Setting Gamma to 0.1
        gamma = 0.1

        for epc in range(self.num_epochs):
            delta_w_sum = 0
            mistake_count = 0
            for i in range(len(features)):
                # Since labels[i] can never be 0, setting to this forces an update if neither work.
                # i.e. when we come across a mistake
                y_prime = 0

                # Using gamma only during training to have a thicker separator.
                if self.sgn_function(np.dot(self.weights, features[i])) > gamma:
                    y_prime = 1
                elif self.sgn_function(np.dot(self.weights, features[i])) < -gamma:
                    y_prime = -1

                if y_prime != labels[i]:
                    # print('Made mistake; epoch no ', epc)
                    mistake_count += 1
                    self.update_weights((self.weights + self.learning_rate*labels[i]*features[i]))
                    
                delta_w_sum += (self.learning_rate*labels[i]*features[i])/self.sample_size

            if np.linalg.norm(delta_w_sum) < 10**-3:
                print("Stopping early on epoch ", epc)
                break
                    # Add tolerance criteria
        return

    def predict(self, data):
        predicted_labels = []
        features, labels = self.feature_extraction(data)
        bias = [[1] for x in range(len(features))]
        features = np.append(features, bias, axis=1)

        for i in range(len(features)):
            prediction = self.sgn_function(np.dot(self.weights,features[i]))
            if prediction == -1:
                prediction = 0
            predicted_labels.append(prediction)
        
        return predicted_labels

def calculate_accuracy(predicted_labels, actual_labels):
    actual_labels = np.array(actual_labels, dtype='float64')
    return 1 - np.sum(np.abs(np.subtract(actual_labels, predicted_labels)))/len(predicted_labels)

def main():
    p = Perceptron()
    train_data = pd.read_csv('trainingSet.csv')
    test_data = pd.read_csv('testSet.csv')
    p.train(train_data)
    train_acc = calculate_accuracy(p.predict(train_data), train_data['decision'])
    test_acc = calculate_accuracy(p.predict(test_data), test_data['decision'])

    print('Train Accuracy for Perceptron: %.2f'%train_acc)
    print('Test Accuracy for Perceptron: %.2f'%test_acc)


if __name__ == '__main__':
    main()