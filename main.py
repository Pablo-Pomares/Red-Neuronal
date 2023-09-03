from mnist_loader import *
import network

training_data, test_data, _ = load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(training_data, 100, 10, 0.05, test_data=test_data)

