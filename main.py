from mnist_loader import *
import network

training_data, test_data, _ = load_data_wrapper()

net = network.Network([784, 30, 10], loss_function="cross_entropy")
net.adam(training_data, 10, 10, 0.001, test_data=test_data)

# with open("net.h", "w") as file:
#    file.write(f"{net.weights},{net.biases}")

