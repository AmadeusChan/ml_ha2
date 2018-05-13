import random
import math

from sklearn.cross_validation import KFold
from sklearn import tree

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
output_path_prefix = "../data/exp2.output"
count_path = "../data/count.txt"

with open(count_path) as f:
    count = int(f.readline())
output_path = output_path_prefix + str(count) + ".csv"

config = {
        "base_model": "d-tree",
        "ensemble": "bagging",
        "T": 10
        }

X, y, X_test = Utils.load_data(train_path, test_path)
y_test = []

cv = 5
kf = KFold(len(X_train), cv)
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
    if config["ensemble"] == "bagging":
        ensemble = Bagging()
    ensemble.aggregate(X_train, y_train, config["T"], base_model, is_classification = True)
    y_valid_ = ensemble.predict(X_valid)
    rmse = cal_rmse(y_valid, y_valid_)
    total_rmse += rmse
    print "valid rmse: %.4f" % (rmse) 

    ensembles.append(ensemble)

print "\naverage rmse: %.4f" % (total_rmse)

y_test = MajorityVoting().aggregate(ensembles).predict(X_test)
with open(output_path) as f:
    f.write("id,label\n")
    for i in range(len(y_test)):
        f.write(str(i + 1) + "," + str(y_test[i]) + "\n")
