from mnist_loader import *
import network

training_data, test_data, _ = load_data_wrapper()

net = network.Network([784, 30, 10])
net.adam(training_data, 10, 10, 0.5, test_data=test_data)

# with open("net.h", "w") as file:
#    file.write(f"{net.weights},{net.biases}")

