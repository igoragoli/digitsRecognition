from loader import load_data
from neural_network import NeuralNetwork

train_set, test_set = load_data()

model = NeuralNetwork(train_set[0], train_set[1], [784, 3, 3, 10])