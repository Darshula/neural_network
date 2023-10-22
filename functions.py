from numpy import exp, power


class Activations:

    @staticmethod
    def linear(x: float) -> float:
        return x

    @staticmethod
    def relu(x: float) -> float:
        return (x > 0) * x

    # @staticmethod
    # def softmax(x: float) -> float:
    #     return (exp(x - 1) /
    #             np.sum(exp(x - 1)))

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + exp(-x))

    @staticmethod
    def tanh(x: float) -> float:
        return (exp(x) - exp(-x) /
                exp(x) + exp(-x))

    # derivatives
    @staticmethod
    def linear_derivative(x: float) -> float:
        return x

    @staticmethod
    def relu_derivative(x: float) -> float:
        return (x > 0) * x

    # @staticmethod
    # def softmax_derivative(x: float) -> float:
    #     exps = exp(x - 1)
    #     return np.divide(exps, np.sum(exps))

    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        return x * (1 - x)

    @staticmethod
    def tanh_derivative(x: float) -> float:
        tanh = (exp(x) - exp(-x) /
                exp(x) + exp(-x))
        return 1 - power(tanh, 2)


class Losses:
    @staticmethod
    def MSE(target: float, activation: float) -> float:
        return power((activation - target), 2) / 2

    @staticmethod
    def CrossEntropy(target: float, activation: float) -> float:
        return 0.0

    # Derivatives
    @staticmethod
    def MSE_derivative(target: float, activation: float) -> float:
        return target - activation

    @staticmethod
    def CrossEntropy_derivative(target: float, activation: float) -> float:
        return 0.0

    if __name__ == '__main__':
        x = 20
        print(Activations.tanh(x), Activations.tanh_derivative(x))
