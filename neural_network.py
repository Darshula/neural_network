from typing import Callable, List

import numpy as np
import numpy.typing as npt

from functions import Activations, Losses
from logger import Logger


class Neuron:
    def __init__(self, index: int, activation_function: str,
                 logger: Logger) -> None:
        self.index = index
        self.logger = logger

        try:
            self.activation_function = getattr(
                Activations, activation_function)
            self.activation_function_derivative = getattr(
                Activations, activation_function+'_derivative')
        except AttributeError:
            self.logger.log_error(
                f'"{activation_function}" is not implemented')

        self.activation: float = 0.0
        self.activation_derivative: float = 0.0
        self.output_synapses: list[Synapse] = []
        self.input_synapses: list[Synapse] = []
        self.error: float = 0.0
        self.error_derivative: float = 0.0

        self.label: str | None = None

    def log_info(self):
        self.logger.log_info(f'Neuron: {self.index}',
                             f'Activation: {self.activation}',
                             f'Label: {self.label}')

    def connect_to(self, receiver) -> None:
        new_connection = Synapse(sender=self, receiver=receiver)
        self.output_synapses.append(new_connection)
        receiver.input_synapses.append(new_connection)

    def forward_pass(self) -> None:
        # summation of x * w of previous layer
        self.raw_input = sum([synapse.weight * synapse.sender.activation
                              for synapse in self.input_synapses])
        self.activation = self.activation_function(self.raw_input)
        self.activation_derivative = self.activation_function_derivative(
            self.activation)

    def backward_pass(self, target: str,
                      loss_function: Callable,
                      loss_derivative: Callable,
                      learning_rate: float,
                      momentum: float) -> None:
        if self.label:
            desired_output = int(target == self.label)
            # MSE
            self.error = loss_function(desired_output, self.activation)
            self.error_derivative = loss_derivative(self.activation,
                                                    desired_output)
        else:
            for connections in self.output_synapses:
                connections.backward_pass(learning_rate, momentum)


class Synapse:
    def __init__(self, sender: Neuron, receiver: Neuron) -> None:
        """Represents the connection between two neurons."""
        self.sender = sender
        self.receiver = receiver
        self.weight: float = 1 - np.random.random()

    def backward_pass(self, learning_rate: float, momentum: float) -> None:
        error = (self.receiver.error_derivative
                 * self.sender.activation_derivative)
        self.sender.error_derivative = (self.receiver.error_derivative
                                        * self.weight
                                        * self.sender.activation_derivative)
        self.weight -= (learning_rate * error * self.sender.activation)


class Layer:
    def __init__(self, neurons: int, activation: str, logger: Logger) -> None:
        """Handles the connection between two neurons."""
        self.logger = logger
        self.neurons = [Neuron(index+1, activation, self.logger)
                        for index in range(neurons)]

    def assign_input(self, inputs: npt.NDArray) -> None:
        for index, neuron in enumerate(self.neurons):
            neuron.activation = neuron.activation_function(inputs[index])

    def assign_labels(self, labels: npt.NDArray) -> None:
        unique_labels = np.unique(labels)
        if len(unique_labels) < len(self.neurons):
            count_unlabeled_neurons = len(labels) - len(self.neurons)
            for index in range(count_unlabeled_neurons, len(self.neurons)):
                self.neurons[index].label = 'Unknown'
        for index in range(len(unique_labels)):
            self.neurons[index].label = unique_labels[index]

    def forward_pass(self) -> None:
        """Applies the activation function over the activations
           of all the neurons in the whole layer"""
        for neuron in self.neurons:
            neuron.forward_pass()

    def backward_pass(self, targets: str,
                      loss_function: Callable,
                      loss_derivative: Callable,
                      learning_rate: float,
                      momentum: float) -> None:
        for neuron in self.neurons:
            neuron.backward_pass(targets,
                                 loss_function,
                                 loss_derivative,
                                 learning_rate,
                                 momentum)

    def predict(self) -> str:
        max_activation: float = 0.0
        prediction: str = ''
        for neuron in self.neurons:
            self.logger.log_info(neuron.activation)
            if max_activation < neuron.activation:
                max_activation = neuron.activation
                prediction = str(neuron.label)
        return prediction


class Network:
    """Represents a fully-connected backpropagation network
       with methods to train and evaluate its performance."""

    def __init__(self, layers: tuple, activations: List[str],
                 logger: Logger | None = None) -> None:
        self.logger = logger if logger else Logger()
        # Same activation will be applied to layers if not specified.
        # The last value is reserved for the output layer.
        if len(layers) != len(activations):
            length = len(activations) - 1
            difference = len(layers) - len(activations)
            for i in range(length, length + difference):
                activations.insert(i, activations[i-1])

        self.layers = [Layer(layers[i], activations[i], self.logger)
                       for i in range(len(layers))]

        # Connecting each neuron of a layer to every neuron of the next layer
        for layer in range(len(self.layers) - 1):
            for sender_neuron in self.layers[layer].neurons:
                for receiver_neuron in self.layers[layer + 1].neurons:
                    sender_neuron.connect_to(receiver_neuron)

        self.logger.log_info('Network Created')

    def forward_pass(self) -> None:
        for layer in self.layers[1::]:
            layer.forward_pass()

    def backward_pass(self, targets: str,
                      loss_function: Callable,
                      loss_derivative: Callable,
                      learning_rate: float,
                      momentum: float) -> None:
        for layer in reversed(self.layers):
            layer.backward_pass(targets,
                                loss_function,
                                loss_derivative,
                                learning_rate,
                                momentum)

    def train(self, data: npt.NDArray, labels: npt.NDArray,
              epochs: int, loss_function: str,
              learning_rate: float, momentum: float = 0.0) -> npt.ArrayLike:

        self.layers[-1].assign_labels(labels)

        try:
            self.loss_function = getattr(
                Losses, loss_function)
            self.loss_derivative = getattr(
                Losses, loss_function+'_derivative')
        except AttributeError:
            self.logger.log_error(
                f'"{loss_function}" is not implemented')

        self.total_error: List[float] = []
        for epoch in range(epochs):
            epoch_error: float = 0.0
            for data_index in range(len(data)):
                self.layers[0].assign_input(data[data_index])

                self.forward_pass()
                self.backward_pass(targets=labels[data_index],
                                   loss_function=self.loss_function,
                                   loss_derivative=self.loss_derivative,
                                   learning_rate=learning_rate,
                                   momentum=momentum)
            for output_neuron in self.layers[-1].neurons:
                epoch_error += output_neuron.error
            self.total_error.append(epoch_error)
            if epoch % 10 == 9:
                self.logger.log_info(
                    f'Epoch {epoch+1}: Error={self.total_error[epoch]}')
                # for i in range(len(self.layers)):
                #     self.logger.log_info(f'Layer: {i + 1}')
                #     for neuron in self.layers[i].neurons:
                #         neuron.log_info()
        self.logger.log_info(
            f'Epoch {epochs}: Error={self.total_error[epochs-1]}')
        for i in range(len(self.layers)):
            self.logger.log_info(f'Layer: {i + 1}')
            for neuron in self.layers[i].neurons:
                neuron.log_info()

        return self.total_error

    def predict(self, data: npt.NDArray) -> str:
        self.layers[0].assign_input(data)
        for layer in self.layers[1::]:
            layer.forward_pass()
        return self.layers[-1].predict()


if __name__ == '__main__':
    # data_sources = {
    #     "training_images": "train-images.idx3-ubyte",
    #     "test_images": "t10k-images.idx3-ubyte",
    #     "training_labels": "train-labels.idx1-ubyte",
    #     "test_labels": "t10k-labels.idx1-ubyte",
    # }

    # mnist: dict[str, np.ndarray] = {}

    # for key in ("training_images", "test_images"):
    #     with open(f'data/{data_sources[key]}', "rb") as mnist_dataset:
    #         mnist[key] = np.frombuffer(
    #             mnist_dataset.read(), np.uint8, offset=16
    #         ).reshape(-1, 28 * 28)

    # for key in ("training_labels", "test_labels"):
    #     with open(f'data/{data_sources[key]}', "rb") as mnist_dataset:
    #         mnist[key] = np.frombuffer(
    #             mnist_dataset.read(), np.uint8, offset=8)
    # x, y, x_test, y_test = (
    #     mnist["training_images"],
    #     mnist["training_labels"],
    #     mnist["test_images"],
    #     mnist["test_labels"],
    # )

    # from sklearn import datasets

    # x, y = datasets.load_digits(return_X_y=True)

    # norm = np.linalg.norm(np.array(x), ord=1)

    # x = np.divide(norm, np.array(x), where=x != 0)
    # x[x == norm] = 0

    # print(x)

    # nn = Network(layers=(64, 32, 16, 10),
    #              activations=['relu', 'sigmoid', 'relu', 'sigmoid'])
    # nn.train(data=np.array(x), labels=np.array(y), loss_function='MSE',
    #          epochs=20, learning_rate=1)

    # print(nn.predict(x[20]), y[20])
    # print(nn.predict(x[200]), y[200])
    # print(nn.predict(x[530]), y[530])
    # print(nn.predict(x[140]), y[140])
    # print(nn.predict(x[763]), y[763])

    logger = Logger()

    from sklearn import datasets

    x, y = datasets.load_iris(return_X_y=True)

    x = np.array(x)
    y = np.array(y)

    nn = Network(layers=(4, 6, 8, 3),
                 activations=['sigmoid', 'relu', 'sigmoid'],
                 logger=logger)

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]

    nn.train(data=x, labels=y, loss_function='MSE',
             epochs=1000, learning_rate=0.01)

    TOTAL_SAMPLES = 10
    correct = 0

    for _ in range(TOTAL_SAMPLES):
        random_sample = int(x.shape[0] * (np.random.random()))
        prediction = nn.predict(x[random_sample])
        answer = y[random_sample]
        logger.log_info(f'{answer} -> {prediction}')
        if int(prediction) == int(answer):
            correct += 1
    logger.log_info(f'Accuracy: {correct} out of {TOTAL_SAMPLES}')
