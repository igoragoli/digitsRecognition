from loader import load_data
from neural_network import NeuralNetwork

train_set, test_set = load_data()
X = train_set[0]
Y = train_set[1]
n0 = X.shape[0]
nL = Y.shape[0]

model = NeuralNetwork(X=X, Y=Y, layers_dims=[n0, 3, 3, nL], learning_rate=0.05, iterations=300)