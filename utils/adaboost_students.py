import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd



class Adaboost():
    def __init__(self, M):
        self.M = M

    def __init_weights(self, N):
        """ initialisation of input variables weights"""
        return np.ones(N) / N

    def update_weights(self, gt, predict, weights, weight_weak_classifiers):
        """ update weights functions DO NOT use loops"""
        error = np.sum(weights * (gt != predict) * weight_weak_classifiers)
        alpha = 0.5 * np.log((1 - error) / error)
        weight_weak_classifiers *= np.exp(alpha * (gt * predict))
        weights *= weight_weak_classifiers
        weights /= np.sum(weights)

    def claculate_error(self, gt, predict, weights):
        """ weak classifier error calculation DO NOT use loops"""
        return np.sum(weights * (gt != predict))

    def claculate_classifier_weight(self, gt, predict, weights):
        """ weak classifier weight calculation DO NOT use loops"""
        error = self.claculate_error(gt, predict, weights)
        return 0.5 * np.log((1 - error) / error)

    def train(self, target, vectors):
        """ train model"""
        N, D = vectors.shape
        self.weak_classifiers = []
        self.alphas = []
        weights = self.__init_weights(N)
        weight_weak_classifiers = np.ones(N)
        for m in range(self.M):
            classifier = DecisionTreeClassifier(max_depth=1)
            classifier.fit(vectors, target, sample_weight=weights)
            predict = classifier.predict(vectors)
            error = self.claculate_error(target, predict, weights)
            alpha = self.claculate_classifier_weight(target, predict, weights)
            self.weak_classifiers.append(classifier)
            self.alphas.append(alpha)
            weight_weak_classifiers = np.exp(-alpha * target * predict)
            weights *= weight_weak_classifiers
            weights_sum = np.sum(weights)
            if np.isinf(weights_sum):
                weights = self.__init_weights(N)
            else:
                weights /= weights_sum

    def get_prediction(self, vectors):
        """ adaboost get prediction """
        predictions = np.array([classifier.predict(vectors) for classifier in self.weak_classifiers])
        return np.sign(np.dot(self.alphas, predictions))