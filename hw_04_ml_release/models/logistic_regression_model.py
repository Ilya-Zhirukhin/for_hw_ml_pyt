from typing import Union
import numpy as np
from easydict import EasyDict
import pickle


class LogReg:

    def __init__(self, cfg: EasyDict, number_classes: int, input_vector_dimension: int):
        self.k = number_classes
        self.d = input_vector_dimension
        self.cfg = cfg
        getattr(self, f'weights_init_{cfg.weights_init_type.name}')(**cfg.weights_init_kwargs)

    def weights_init_normal(self, sigma):
        self.W = np.random.normal(scale=sigma, size=(self.k, self.d))
        self.b = np.random.normal(scale=sigma, size=(self.k, 1))

    def weights_init_uniform(self, epsilon):
        self.W = np.random.uniform(-epsilon, epsilon, size=(self.k, self.d))
        self.b = np.random.uniform(-epsilon, epsilon, size=(self.k, 1))

    def weights_init_xavier(self, n_in, n_out):
        scale = np.sqrt(2 / (n_in + n_out))
        self.W = np.random.normal(scale=scale, size=(self.k, self.d))
        self.b = np.random.normal(scale=scale, size=(self.k, 1))

    def weights_init_he(self, n_in):
        scale = np.sqrt(2 / n_in)
        self.W = np.random.normal(scale=scale, size=(self.k, self.d))
        self.b = np.random.normal(scale=scale, size=(self.k, 1))

    def __softmax(self, model_output: np.ndarray) -> np.ndarray:
        exp_z = np.exp(model_output - np.max(model_output, axis=0))
        return exp_z / exp_z.sum(axis=0)

    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        # calculate model confidence (y in lecture)
        z = self.__get_model_output(inputs)
        y = self.__softmax(z)
        return y

    def __get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        return self.W @ inputs.T + self.b

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        return (model_confidence - targets.T) @ inputs / inputs.shape[0]

    def __get_gradient_b(self, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        return np.mean(model_confidence - targets.T, axis=1, keepdims=True)

    def __weights_update(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        grad_w = self.__get_gradient_w(inputs, targets, model_confidence)
        grad_b = self.__get_gradient_b(targets, model_confidence)
        self.W -= self.cfg.gamma * grad_w
        self.b -= self.cfg.gamma * grad_b

    def __gradient_descent_step(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                epoch: int, inputs_valid: Union[np.ndarray, None] = None,
                                targets_valid: Union[np.ndarray, None] = None):
        model_confidence = self.get_model_confidence(inputs_train)
        target_value = self.__target_function_value(inputs_train, targets_train, model_confidence)
        self.__weights_update(inputs_train, targets_train, model_confidence)

        if inputs_valid is not None and targets_valid is not None:
            valid_confidence = self.get_model_confidence(inputs_valid)
            valid_target_value = self.__target_function_value(inputs_valid, targets_valid, valid_confidence)
            self.__validate(inputs_valid, targets_valid, valid_confidence)

    def gradient_descent_epoch(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None):
        # loop stopping criteria - number of iterations of gradient_descent
        for epoch in range(self.cfg.nb_epoch):
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)

    def gradient_descent_gradient_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                       inputs_valid: Union[np.ndarray, None] = None,
                                       targets_valid: Union[np.ndarray, None] = None):
        epoch = 0
        while True:
            model_confidence = self.get_model_confidence(inputs_train)
            grad_w = self.__get_gradient_w(inputs_train, targets_train, model_confidence)
            grad_b = self.__get_gradient_b(targets_train, model_confidence)

            if np.linalg.norm(grad_w) < self.cfg.tol and np.linalg.norm(grad_b) < self.cfg.tol:
                break

            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)
            epoch += 1

    def gradient_descent_difference_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                         inputs_valid: Union[np.ndarray, None] = None,
                                         targets_valid: Union[np.ndarray, None] = None):
        epoch = 0
        while True:
            old_W = self.W.copy()
            old_b = self.b.copy()

            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)

            if np.linalg.norm(self.W - old_W) < self.cfg.tol and np.linalg.norm(self.b - old_b) < self.cfg.tol:
                break

            epoch += 1

    def gradient_descent_metric_value(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                      inputs_valid: Union[np.ndarray, None] = None,
                                      targets_valid: Union[np.ndarray, None] = None):
        epoch = 0
        best_accuracy = -1
        patience = self.cfg.patience
        while patience > 0:
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)
            _, model_confidence = self.__validate(inputs_valid, targets_valid)
            accuracy = self.__calculate_accuracy(targets_valid, model_confidence)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience = self.cfg.patience
            else:
                patience -= 1

            epoch += 1
    def __calculate_accuracy(self, targets: np.ndarray, model_confidence: np.ndarray) -> float:
        predictions = np.argmax(model_confidence, axis=0)
        ground_truth = np.argmax(targets.T, axis=0)
        return np.mean(predictions == ground_truth)

    def train(self, inputs_train: np.ndarray, targets_train: np.ndarray,
              inputs_valid: Union[np.ndarray, None] = None, targets_valid: Union[np.ndarray, None] = None):
        getattr(self, f'gradient_descent_{self.cfg.gd_stopping_criteria.name}')(inputs_train, targets_train,
                                                                                inputs_valid,
                                                                                targets_valid)

    def __target_function_value(self, inputs: np.ndarray, targets: np.ndarray,
                                model_confidence: Union[np.ndarray, None] = None) -> float:
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        log_likelihood = -np.sum(targets.T * np.log(model_confidence + 1e-8))
        return log_likelihood / inputs.shape[0]

    def __validate(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: Union[np.ndarray, None] = None):
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        ground_truth = np.argmax(targets.T, axis=0)
        accuracy = np.mean(predictions == ground_truth)

        confusion_matrix = np.zeros((self.k, self.k), dtype=int)
        for i in range(len(predictions)):
            confusion_matrix[ground_truth[i], predictions[i]] += 1

        return accuracy, confusion_matrix

    def predict(self, inputs):
        return np.argmax(self.forward(inputs), axis=1)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def __call__(self, inputs: np.ndarray):
        model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        return predictions




