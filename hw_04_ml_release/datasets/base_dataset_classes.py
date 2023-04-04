from abc import ABC, abstractmethod
import numpy as np

class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass

    @property
    @abstractmethod
    def d(self):
        # inputs variables
        pass

    def divide_into_sets(self):
        # divide data into training, validation, and test sets
        n = len(self.inputs)
        n_train = int(n * self.train_set_percent)
        n_valid = int(n * self.valid_set_percent)
        n_test = n - n_train - n_valid

        indices = np.random.permutation(n)
        self.inputs_train = self.inputs[indices[:n_train]]
        self.targets_train = self.targets[indices[:n_train]]
        self.inputs_valid = self.inputs[indices[n_train:n_train+n_valid]]
        self.targets_valid = self.targets[indices[n_train:n_train+n_valid]]
        self.inputs_test = self.inputs[indices[-n_test:]]
        self.targets_test = self.targets[indices[-n_test:]]

    def normalization(self):

        self.inputs_train = (self.inputs_train - np.min(self.inputs_train, axis=0)) / np.ptp(self.inputs_train,
                                                                                             axis=0)
        self.inputs_valid = (self.inputs_valid - np.min(self.inputs_valid, axis=0)) / np.ptp(self.inputs_valid,
                                                                                             axis=0)
        self.inputs_test = (self.inputs_test - np.min(self.inputs_test, axis=0)) / np.ptp(self.inputs_test, axis=0)

    def get_data_stats(self):
        # calculate mean and std of inputs vectors of training set by each dimension
        self.mean = np.mean(self.inputs_train, axis=0)
        self.std = np.std(self.inputs_train, axis=0)

    def standardization(self):
        # standardize inputs based on mean and std calculated in get_data_stats()
        self.inputs_train = (self.inputs_train - self.mean) / (self.std + 1e-8)
        self.inputs_valid = (self.inputs_valid - self.mean) / (self.std + 1e-8)
        self.inputs_test = (self.inputs_test - self.mean) / (self.std + 1e-8)

class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        # create matrix of one-hot encoding vectors for input targets
        return np.eye(number_classes)[targets]
