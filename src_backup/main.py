import random
import math

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

train_path = "../data/exp2.train.csv"
test_path = "../data/exp2.validation_review.csv"
output_path_prefix = "../result/exp2.output"
count_path = "../data/count.txt"

with open(count_path, "r") as f:
    count = int(f.readline())
output_path = output_path_prefix + str(count) + ".csv"
count += 1
with open(count_path, "w") as f:
    f.write(str(count) + "\n")

config = {
        "base_model": "svm",
        "ensemble": "adaboosting",
        "T": 5
        }

X, y, X_test = Utils.load_data(train_path, test_path)
X, y, X_test = np.asarray(X), np.asarray(y), np.asarray(X_test)
y_test = []

cv = 5
kf = KFold(len(X), cv)
ensembles = []
total_rmse = 0.

cnt = 0
for train_idx, valid_idx in kf:
    print "\n************************* Fold %s *************************" % (cnt)
    cnt += 1
    X_train, y_train = X[train_idx], y[train_idx]  
    X_valid, y_valid= X[valid_idx], y[valid_idx]    

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
