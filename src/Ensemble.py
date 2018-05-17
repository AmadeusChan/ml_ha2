import numpy as np
import sklearn
import unittest
import random
import copy
import math

class Ensemble(object):

    def __init__(self):
        pass

    def aggregate(self):
        assert False

    def predict(self, X_test):
        assert False

class MajorityVoting(Ensemble):

    def __init__(self):
        self.classifiers = []

    def aggregate(self, classifiers):
        self.classifiers = classifiers
        return self

    def predict(self, X_test):
        y_test_ = []
        for classifier in self.classifiers:
            temp = classifier.predict(X_test)
            y_test_.append(temp)
        y_test = []
        for i in range(len(X_test)):
            seen = {}
            for j in range(len(self.classifiers)):
                if y_test_[j][i] in seen:
                    seen[y_test_[j][i]] += 1
                else:
                    seen[y_test_[j][i]] = 1
            max_val = 0
            result = None
            for j in seen.keys():
                if seen[j] > max_val:
                    max_val = seen[j]
                    result = j
            y_test.append(result)
        return y_test

class WeightVoting(Ensemble):

    def __init__(self):
        self.classifiers = []
        self.weights = []

    def aggregate(self, classifiers, weights):
        self.classifiers = classifiers
        self.weights = weights
        return self
 
    def predict(self, X_test):
        y_test_ = []
        for classifier in self.classifiers:
            temp = classifier.predict(X_test)
            y_test_.append(temp)
        y_test = []
        for i in range(len(X_test)):
            seen = {}
            for j in range(len(self.classifiers)):
                if y_test_[j][i] in seen:
                    seen[y_test_[j][i]] += self.weights[j]
                else:
                    seen[y_test_[j][i]] = self.weights[j]
            max_val = -1e20
            result = None
            for j in seen.keys():
                if seen[j] > max_val:
                    max_val = seen[j]
                    result = j
            y_test.append(result)
        return y_test
   
class SimpleAveraging(Ensemble):
    
    def __init__(self):
        self.classifiers = []

    def aggregate(self, classifiers):
        self.classifiers = classifiers
        return self

    def predict(self, X_test):
        y_test_ = []
        for classifier in self.classifiers:
            temp = classifier.predict(X_test)
            y_test_.append(temp)
        y_test = []
        for i in range(len(X_test)):
            result = 0.
            for j in range(len(self.classifiers)):
                result += y_test_[j][i]
            result /= float(len(self.classifiers))
            y_test.append(result)
        return y_test

class Bagging(Ensemble):

    def __init__(self):
        self.base_model = None
        self.is_classification = None
        self.classifiers = None
        self.combining_strategy = None

    def bootstrap(self, X_train, y_train):
        X_train_ = []
        y_train_ = []
        for i in range(len(X_train)):
            idx = random.randint(0, len(X_train) - 1)
            X_train_.append(X_train[idx])
            y_train_.append(y_train[idx])
        return X_train_, y_train_

    def config(self, T, base_model, is_classification):
        self.base_model = copy.deepcopy(base_model)
        self.is_classification = is_classification
        self.classifiers = []
        if is_classification:
            self.combining_strategy = MajorityVoting()
        else:
            self.combining_strategy = SimpleAveraging()
        return self

    def fit(self, X_train, y_train):
        for t in range(T):
            print "bagging t = %s/%s" % (t + 1, T)
            classifier = copy.deepcopy(self.base_model)
            X_train_, y_train_ = self.bootstrap(X_train, y_train)
            classifier.fit(X_train_, y_train_)
            self.classifiers.append(classifier)
        self.combining_strategy.aggregate(self.classifiers)
        return self

    def predict(self, X_test):
        assert self.base_model != None
        assert self.is_classification != None
        assert self.combining_strategy != None
        assert self.combining_strategy != None
        return self.combining_strategy.predict(X_test)

class AdaBoostingM1(Ensemble):

    def __init__(self):
        self.base_model = None
        self.is_classification = None
        self.classifiers = None
        self.combining_strategy = None
        self.alpha = None

    # default time = 3, which means len(X_train_) = 3 x len(X_train)
    # since lim_{m->inf} (1-1/m)^(3m) \approx 0.05, which means on average over 95% trainning data are covered
    def weighted_sampling(self, X_train, y_train, weights, times = 3):
        assert len(X_train) > 0
        X_train_ = []
        y_train_ = []
        accumulated_weights = []
        s = 0.
        for i in range(len(weights)):
            s += weights[i]
            accumulated_weights.append(s)
        #print "%.10f\n" % (accumulated_weights[-1])
        assert abs(accumulated_weights[-1] - 1.) < 1e-8, "weights are not normalized"
        N = len(X_train)
        for i in range(N * times):
            r = random.uniform(0., 1.)
            idx = None
            if r <= accumulated_weights[0]:
                idx = 0
            elif r > accumulated_weights[-2]:
                idx = N - 1
            else:
                left = 0
                right = N - 1
                while right - left > 1:
                    mid = (left + right) / 2
                    if r > accumulated_weights[mid - 1] and r <= accumulated_weights[mid]:
                        idx = mid
                        break
                    elif r <= accumulated_weights[mid - 1]:
                        right = mid
                    else:
                        left = mid
            assert idx != None
            X_train_.append(X_train[idx])
            y_train_.append(y_train[idx])
        return X_train_, y_train_

    def config(self, T, base_model, is_classification):
        self.base_model = copy.deepcopy(base_model)
        self.is_classification = is_classification
        self.classifiers = []
        self.alpha = []
        if is_classification:
            self.combining_strategy = WeightVoting()
        else:
            assert False # AdaBoosting for regression has not been supported yet
            self.combining_strategy = SimpleAveraging()
        return self

    def fit(self, X_train, y_train):
        weights = []
        N = len(X_train)
        for i in range(len(X_train)):
            weights.append(1. / N)

        for t in range(T):
            print "adaboosting t = %s/%s" % (t + 1, T)
            classifier = copy.deepcopy(self.base_model)
            X_train_, y_train_ = self.weighted_sampling(X_train, y_train, weights)
            classifier.fit(X_train_, y_train_)
            predicted_y_train = classifier.predict(X_train)
            epsilon = 0.
            for i in range(len(X_train)):
                if y_train[i] != predicted_y_train[i]:
                    epsilon += weights[i]
            if epsilon > 0.5:
                break
            beta = epsilon / (1. - epsilon)
            weight_sum = 0.
            for i in range(len(X_train)):
                if y_train[i] == predicted_y_train[i]:
                    weights[i] *= beta
                weight_sum += weights[i]
            # normalization
            for i in range(len(X_train)):
                weights[i] /= weight_sum

            self.alpha.append(math.log(1. / beta))
            self.classifiers.append(classifier)
        self.combining_strategy.aggregate(self.classifiers, self.alpha)
        return self

    def predict(self, X_test):
        assert self.base_model != None
        assert self.is_classification != None
        assert self.combining_strategy != None
        assert self.combining_strategy != None
        assert self.alpha != None
        return self.combining_strategy.predict(X_test)

class Stacking(Ensemble):

    def __init__(self):
        pass

    def aggregate(self):
        pass

    def predict(self, X_test):
        pass

class FakeClassifier(object):

    def __init__(self):
        self.fixed_output = None

    def train(self, fixed_output):
        self.fixed_output = fixed_output
        return self

    def predict(self, X_test):
        y_test = []
        for i in range(len(X_test)):
            y_test.append(self.fixed_output)
        return y_test

class EnsembleTestCase(unittest.TestCase):

    def test_majority_voting(self):
        vote = MajorityVoting()
        cfs = [FakeClassifier().train(0), FakeClassifier().train(1), FakeClassifier().train(1), FakeClassifier().train(3)]
        X_test = [ [0, 0] ] * 10
        vote.aggregate(cfs)
        y_test = vote.predict(X_test)
        assert len(y_test) == 10
        for i in range(len(y_test)):
            assert y_test[i] == 1

    def test_simple_averaging(self):
        vote = SimpleAveraging()
        cfs = [FakeClassifier().train(0), FakeClassifier().train(1), FakeClassifier().train(1), FakeClassifier().train(3)]
        X_test = [ [0, 0] ] * 10
        vote.aggregate(cfs)
        y_test = vote.predict(X_test)
        assert len(y_test) == 10
        for i in range(len(y_test)):
            assert abs(y_test[i] - 1.25) < 1e-20

    def test_weight_voting(self):
        vote = WeightVoting()
        cfs = [FakeClassifier().train(0), FakeClassifier().train(1), FakeClassifier().train(1), FakeClassifier().train(3)]
        X_test = [ [0, 0] ] * 10
        weights = [1., 1., 1., 10.]
        vote.aggregate(cfs, weights)
        y_test = vote.predict(X_test)
        assert len(y_test) == 10
        for i in range(len(y_test)):
            assert y_test[i] == 3

if __name__ == "__main__":
    unittest.main()
