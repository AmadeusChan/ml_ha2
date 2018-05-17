import random
import math
import copy

import numpy as np

from sklearn.cross_validation import KFold
from sklearn import tree
from sklearn.svm import LinearSVC

import Utils
from Ensemble import *

def cal_rmse(y_true, y_predict):
    loss = 0.
    N = len(y_true)
    for i in range(N):
        loss += (y_true[i] - y_predict[i]) ** 2
    loss = math.sqrt(loss * 1. / N)
    return loss

def data_augmentation(X_train, y_train):
    X_train_ = []
    y_train_ = []
    N = len(X_train)
    for i in range(N):
        x = copy.deepcopy(X_train[i])
        y = copy.deepcopy(y_train[i])
        X_train_.append(x)
        y_train_.append(y)
        x = copy.deepcopy(x)
        x[random.randint(0, len(x) - 1)] ^= 1
        X_train_.append(x)
        y_train_.append(y)
    idx = range(len(X_train_))
    random.shuffle(idx)
    X_train_, y_train_ = np.asarray(X_train_), np.asarray(y_train_)
    #X_train_, y_train_ = X_train_[idx], y_train_[idx]
    return X_train_, y_train_

train_path = "../data/exp2.train.csv"
test_path = "../data/exp2.validation_review.csv"
output_path_prefix = "../result/exp2.output"
count_path = "../data/count.txt"

with open(count_path, "r") as f:
    count = int(f.readline())
count += 1
with open(count_path, "w") as f:
    f.write(str(count) + "\n")

config = {
        "base_model": "d-tree",
        "ensemble": "adaboosting",
        "T": 10
        "comment": "large_T_large_cv"
        }

cv = 10
output_path = output_path_prefix + "_cv_" + str(cv) + "_comment_" + config["comment"] + "_base_model_" + config["base_model"] + "_ensemble_" + config["ensemble"] + "_T_" + str(config["T"]) + "_" + str(count) + ".csv"

X, y, X_test = Utils.load_data(train_path, test_path)
X, y, X_test = np.asarray(X), np.asarray(y), np.asarray(X_test)
y_test = []

kf = KFold(len(X), cv)
ensembles = []
total_rmse = 0.

cnt = 0
for train_idx, valid_idx in kf:
    print "\n************************* Fold %s *************************" % (cnt)
    cnt += 1
    X_train, y_train = X[train_idx], y[train_idx]  
    X_valid, y_valid= X[valid_idx], y[valid_idx]    
    """
    print "size of training set before augmentation: %s" % (len(X_train))
    X_train, y_train = data_augmentation(X_train, y_train)
    print "size of training set after augmentation: %s" % (len(X_train))
    """

    if config["base_model"] == "d-tree":
        base_model = tree.DecisionTreeClassifier()
    elif config["base_model"] == "svm":
        base_model = LinearSVC(random_state=0)

    if config["ensemble"] == "bagging":
        ensemble = Bagging()
    else:
        ensemble = AdaBoostingM1()

    ensemble.aggregate(X_train, y_train, config["T"], base_model, is_classification = True)
    y_valid_ = ensemble.predict(X_valid)

    """
    cf = tree.DecisionTreeClassifier()
    cf.fit(X_train, y_train)
    y_valid_ = cf.predict(X_valid)
    """

    rmse = cal_rmse(y_valid, y_valid_)
    total_rmse += rmse
    print "valid rmse: %.4f" % (rmse) 

    ensembles.append(ensemble)
    #ensembles.append(cf)

print "\naverage rmse: %.4f" % (total_rmse / cv)

y_test = MajorityVoting().aggregate(ensembles).predict(X_test)
with open(output_path, "w") as f:
    f.write("id,label\n")
    for i in range(len(y_test)):
        f.write(str(i + 1) + "," + str(y_test[i]) + "\n")
