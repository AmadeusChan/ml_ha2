import numpy as np
import random
import re
import sys
import json
import os

def extract_bitrigram(raw_X, y, bigram_path, trigram_path):
    bigram_list = None
    trigram_list = None

    bigram_th = 5e-5
    trigram_th = 5e-5

    if os.path.isfile(bigram_path):
        with open(bigram_path, "r") as f:
            bigram_list = json.load(f)
    else:
        print ""
        bigrams = {}
        tot_count = 0 
        for i in range(len(raw_X)):
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print "extracting bigrams... %.2f" % (100. * (i + 1.) / len(raw_X)) + "%"

            x = raw_X[i]
            if len(x) - 1 <= 0:
                continue
            for j in range(len(x) - 1):
                tot_count += 1
                w = x[j] + "100001" + x[j + 1]
                if w in bigrams:
                    bigrams[w][y[i]] += 1
                    bigrams[w]["count"] += 1
                else:
                    stat = {0: 0, -1: 0, 1: 0, "count": 1}
                    stat[y[i]] = 1 
                    bigrams[w] = stat
        count_th = int(bigram_th * tot_count)
        bigram_list = []
        print "total number of bigrams %s" % (len(bigrams))
        for w in bigrams:
            stat = bigrams[w]
            if stat["count"] < count_th:
                continue
            stat["word"] = w
            bigram_list.append(stat)
        print "number of bigrams selected %s" % (len(bigram_list))
        with open(bigram_path, "w") as f:
            json.dump(bigram_list, f)
    
    if os.path.isfile(trigram_path):
        with open(trigram_path, "r") as f:
            trigram_list = json.load(f)
    else:
        print ""
        trigrams = {}
        tot_count = 0 
        for i in range(len(raw_X)):
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print "extracting trigrams... %.2f" % (100. * (i + 1.) / len(raw_X)) + "%"

            x = raw_X[i]
            if len(x) - 2 <= 0:
                continue
            for j in range(len(x) - 2):
                tot_count += 1
                w = x[j] + "100001" + x[j + 1] + "100001" + x[j + 2]
                if w in trigrams:
                    trigrams[w][y[i]] += 1
                    trigrams[w]["count"] += 1
                else:
                    stat = {0: 0, -1: 0, 1: 0, "count": 1}
                    stat[y[i]] = 1 
                    trigrams[w] = stat
        count_th = int(trigram_th * tot_count)
        trigram_list = []
        print "total number of trigrams %s" % (len(trigrams))
        for w in trigrams:
            stat = trigrams[w]
            if stat["count"] < count_th:
                continue
            stat["word"] = w
            trigram_list.append(stat)
        print "number of trigrams selected %s" % (len(trigram_list))
        with open(trigram_path, "w") as f:
            json.dump(trigram_list, f)

    return trigram_list, trigram_list

def extract_keywords(row_X, y, keywords_path):
    if os.path.isfile(keywords_path):
        with open(keywords_path, "r") as f:
            keywords_list = json.load(f)
        return keywords_list

    words = {}
    print ""
    for i in range(len(row_X)):
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
        print "extracting keywords... %.2f" % (100. * (i + 1.) / len(row_X)) + "%"

        x = row_X[i]
        for j in range(len(x)):
            w = x[j]
            if w in words:
                words[w][y[i]] += 1
            else:
                stat = {0: 0, -1: 0, 1: 0}
                stat[y[i]] = 1
                words[w] = stat
    print "number of keywords: %s" %(len(words))

    words_list = []
    for word in words:
        stat = words[word]
        stat["word"] = word
        stat["importance"] = max(max(stat[0], stat[1]), stat[-1]) * 1. / (stat[0] + stat[1] + stat[-1])
        stat["count"] = stat[0] + stat[1] + stat[-1]
        """
        if (stat[0] + stat[1] + stat[-1]) < 20:
            stat["importance"] = 0.
        """
        words_list.append(stat)

    print ""
    for i in range(len(words_list)):
    #for i in range(500):
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
        print "sorting keywords... %.2f" % (100. * (i + 1.) / len(words_list)) + "%"

        for j in range(i, len(words_list)):
            if words_list[i]["count"] < words_list[j]["count"]:
            #if words_list[i]["importance"] < words_list[j]["importance"]:
                words_list[i], words_list[j] = words_list[j], words_list[i]

    top_number = len(words_list)
    #top_number = 17000
    keywords_list = []
    for i in range(top_number):
        print "ranking " + str(i) + ": " + words_list[i]["word"] + " score: %.4f" % (words_list[i]["importance"])
        keywords_list.append(words_list[i])

    with open(keywords_path, "w") as f:
        json.dump(keywords_list, f)

    return keywords_list

def process_X(line): 
    return re.split(r" +", line.strip())

def load_data(train_path, test_path):
    print "loading data..."
    
    keywords_path = "../data/keywords.json"
    X_path = "../data/X.npy"
    X_test_path = "../data/X_test.npy"
    bigram_path = "../data/bigrams.json"
    trigram_path = "../data/trigrams.json"

    raw_X = []
    X = []
    y = []
    raw_X_test = []
    X_test = []

    with open(train_path, "r") as f:
        line = f.readline()
        while True:
            line = f.readline().strip()
            if line == None or len(line) == 0:
                break
            pos = 0
            while line[pos] != ',':
                pos += 1
            y.append(int(line[:pos]))
            raw_X.append(process_X(line[pos+1:]))

    bigram_list, trigram_list = extract_bitrigram(raw_X, y, bigram_path, trigram_path)

    keywords_list = extract_keywords(raw_X, y, keywords_path)
    keywords_set = set()
    for keyword in keywords_list:
        keywords_set.add(keyword["word"])

    if os.path.isfile(X_path):
        X = np.load(X_path)
    else:
        print ""
        for i in range(len(raw_X)):
        #for i in range(500):
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print "generating training set features... %.2f" % ((1. + i) / len(raw_X) * 100.) + "%"
    
            x = []
            word_set = set()
            for w in raw_X[i]:
                word_set.add(w)
            for j in range(len(keywords_list)):
                w_ = keywords_list[j]["word"]
                try:
                    _ = w_.encode('utf-8')
                    w_ = _
                except:
                    pass
                if w_ in word_set:
                    x.append(1)
                else:
                    x.append(0)

            bigram_set = set()
            if len(raw_X[i]) - 1 > 0:
                for j in range(len(raw_X[i]) - 1):
                    w = raw_X[i][j] + "100001" + raw_X[i][j + 1]
                    bigram_set.add(w)
            for j in range(len(bigram_list)):
                w_ = bigram_list[j]["word"]
                try:
                    _ = w_.encode('utf-8')
                    w_ = _
                except:
                    pass
                if w_ in bigram_set:
                    x.append(1)
                else:
                    x.append(0)

            trigram_set = set()
            if len(raw_X[i]) - 2 > 0:
                for j in range(len(raw_X[i]) - 2):
                    w = raw_X[i][j] + "100001" + raw_X[i][j + 1] + "100001" + raw_X[i][j + 2]
                    trigram_set.add(w)
            for j in range(len(trigram_list)):
                w_ = trigram_list[j]["word"]
                try:
                    _ = w_.encode('utf-8')
                    w_ = _
                except:
                    pass
                if w_ in trigram_set:
                    x.append(1)
                else:
                    x.append(0)

            x.append(len(raw_X[i]))
            X.append(x)
        np.save(X_path, np.asarray(X))
    
    with open(test_path) as f:
        line = f.readline()
        while True:
            line = f.readline().strip()
            if line == None or len(line) == 0:
                break
            pos = 0
            while line[pos] != ',':
                pos += 1
            raw_X_test.append(process_X(line[pos+1:]))

    if os.path.isfile(X_test_path):
        X_test = np.load(X_test_path)
    else:
        print ""
        #for i in range(500):
        for i in range(len(raw_X_test)):
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print "generating test set features... %.2f" % ((1. + i) / len(raw_X_test) * 100.) + "%"
    
            x = []
            word_set = set()
            for w in raw_X_test[i]:
                word_set.add(w)
            for j in range(len(keywords_list)):
                w_ = keywords_list[j]["word"]
                try:
                    _ = w_.encode('utf-8')
                    w_ = _
                except:
                    pass
                if w_ in word_set:
                    x.append(1)
                else:
                    x.append(0)

            bigram_set = set()
            if len(raw_X_test[i]) - 1 > 0:
                for j in range(len(raw_X_test[i]) - 1):
                    w = raw_X_test[i][j] + "100001" + raw_X_test[i][j + 1]
                    bigram_set.add(w)
            for j in range(len(bigram_list)):
                w_ = bigram_list[j]["word"]
                try:
                    _ = w_.encode('utf-8')
                    w_ = _
                except:
                    pass
                if w_ in bigram_set:
                    x.append(1)
                else:
                    x.append(0)

            trigram_set = set()
            if len(raw_X_test[i]) - 2 > 0:
                for j in range(len(raw_X_test[i]) - 2):
                    w = raw_X_test[i][j] + "100001" + raw_X_test[i][j + 1] + "100001" + raw_X_test[i][j + 2]
                    trigram_set.add(w)
            for j in range(len(trigram_list)):
                w_ = trigram_list[j]["word"]
                try:
                    _ = w_.encode('utf-8')
                    w_ = _
                except:
                    pass
                if w_ in trigram_set:
                    x.append(1)
                else:
                    x.append(0)

            x.append(len(raw_X_test[i]))
            X_test.append(x)
        np.save(X_test_path, np.asarray(X_test))
    
    X, y, X_test = np.asarray(X), np.asarray(y), np.asarray(X_test)
    idx = range(len(X))
    random.shuffle(idx)
    X, y = X[idx], y[idx]
    return X, y, X_test

if __name__ == "__main__":
    train_path = "../data/exp2.train.csv"
    test_path = "../data/exp2.validation_review.csv"
    #load_data(train_path, test_path)
    X, y, X_test = load_data(train_path, test_path)
    for i in range(10):
        print X_test[i]
