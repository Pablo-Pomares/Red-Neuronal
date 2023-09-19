"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes, loss_function="mean_square_avg"):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.loss_function = loss_function
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        def update_mini_batch(mini_batch, eta, mini_batch_size):
            """Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
            The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
            is the learning rate."""
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y, mini_batch_size)
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w-(eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]
            
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                update_mini_batch(mini_batch, eta, mini_batch_size)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")
        print("Training complete \(^◇^)/")
            
    def adam(self, training_data, epochs, mini_batch_size, eta, test_data=None, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
        """
        Implementación del optimizador Adam 
        """
        self.mini_batch_size = mini_batch_size 
        
        def update_mini_batch(mini_batch, eta, beta_1, beta_2, epsilon, t, mini_batch_size):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # Llamo a las variables m y v que están fuera de esta función y las actualizo
            nonlocal m_b
            nonlocal m_w
            nonlocal v_b
            nonlocal v_w
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y, mini_batch_size)
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
            m_b = [beta_1*mb + (1-beta_1)*nb for mb, nb in zip(m_b, nabla_b)]
            v_b = [beta_2*vb + (1-beta_2)*(nb**2) for vb, nb in zip(v_b, nabla_b)]
            m_w = [beta_1*mw + (1-beta_1)*nw for mw, nw in zip(m_w, nabla_w)]
            v_w = [beta_2*vw + (1-beta_2)*(nw**2) for vw, nw in zip(v_w, nabla_w)]
            
            # creo las hats de cada m y v
            m_b_hat = [mb/(1-beta_1**t) for mb in m_b]
            m_w_hat = [mw/(1-beta_1**t) for mw in m_w]
            v_b_hat = [vb/(1-beta_2**t) for vb in v_b]
            v_w_hat = [vw/(1-beta_2**t) for vw in v_w]
            
            # Actualizo las w y b 
            self.weights = [w - (eta/(np.sqrt(vw)+epsilon))*mw 
                            for w, vw, mw in zip(self.weights, v_w_hat, m_w_hat)]
            self.biases = [b - (eta/(np.sqrt(vb)+epsilon))*mb 
                            for b, vb, mb in zip(self.biases, v_b_hat, m_b_hat)]
        
            
        if test_data: n_test = len(test_data)
        n = len(training_data)
        m_b = [abs(np.zeros(b.shape)) for b in self.biases] # m para las b
        v_b = [abs(np.zeros(b.shape)) for b in self.biases] # v para las b
        m_w = [abs(np.zeros(w.shape)) for w in self.weights] # m para las w
        v_w = [abs(np.zeros(w.shape)) for w in self.weights] # v para las w
        t = 0
        for j in range(epochs):
            t += 1
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                update_mini_batch(mini_batch, eta, beta_1, beta_2, epsilon, t, mini_batch_size)
            if test_data:
                if self.softmax:
                    print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
                else:
                    print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")
        print("Training complete \(^◇^)/")
            
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def backprop(self, x, y, mini_batch_size):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        match self.loss_function:
            case "mean_square_avg":      
                delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
            case "cross_entropy":
                delta = activations[-1] - y
            case _:
                print("????")
                exit()
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def softmax(self, x):
        '''Implementación de softmax'''
        z = x
        a = [zi/sum(z) for zi in z]
        return a
    
    def softmax_prime(self, x):
        sftmax = self.softmax(x)
        return [ai*(1-ai) for ai in sftmax]
    
    def cross_entropy_derivative(self, x, y):
        """Derivada de cross entropy"""
        a = self.softmax(x)
        nabla_ce = [-yi/(ai) for ai, yi in zip(a, y)]
        return nabla_ce

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))