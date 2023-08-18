import sys
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import utils

import json


def get_data_train_tst(train_csv, tst_csv, cols, trh, target):
    
    print('Descriptors: {}'.format(cols))
    
    train_df = pd.read_csv(train_csv)
    tst_df = pd.read_csv(tst_csv)
    
    X_train = train_df[cols].to_numpy()
    _Y = train_df[target]
    Y_train = np.array([1 if _> trh else 0 for _ in _Y])
    
    X_eval = tst_df[cols].to_numpy()
    _Y = tst_df[target].to_numpy()
    Y_eval = np.array([1 if _> trh else 0 for _ in _Y])
    
    scaler = StandardScaler()
    # Fit StandardScaler on the training set
    scaler.fit(X_train)
    # Normalize tarining and test sets
    X_train = scaler.transform(X_train)
    X_eval = scaler.transform(X_eval)

    return X_train, Y_train, X_eval, Y_eval 

class Metrics():
    
    def __init__(self):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.total_pos_tst = 0
        self.total_neg_tst = 0
        self.total_pos_pred = 0
        self.total_neg_pred = 0
        self.total_acc = 0.0
        self.model = None

        self.precision = 0.0
        self.recal = 0.0
        self.F1 = 0.0


    def compute_metrics(self, model, X_tst, Y_tst):
        Y_pred = model.predict(X_tst)
        
        #print("ActualLabels={}".format(Y_tst.tolist()))
        #print("PredictedLabels={}".format(Y_pred.tolist()))
        
        num_true_neg, num_true_pos = 0, 0
        num_false_neg, num_false_pos = 0, 0

        for k in range(Y_tst.shape[0]):
            if Y_tst[k] == 0 and Y_pred[k] == 0:
                num_true_neg += 1
            elif Y_tst[k] == 1 and Y_pred[k] == 1:
                num_true_pos += 1
            elif Y_tst[k] == 1 and Y_pred[k] == 0:
                num_false_neg += 1
            elif Y_tst[k] == 0 and Y_pred[k] == 1:
                num_false_pos += 1
        
        self.model = model

        self.true_pos = num_true_pos
        self.true_neg = num_true_neg
        self.false_pos = num_false_pos
        self.false_neg = num_false_neg
       
        self.total_neg_tst = (Y_tst == 0).sum()
        self.total_pos_tst = (Y_tst == 1).sum()
        self.total_neg_pred = (Y_pred == 0).sum()
        self.total_pos_pred = (Y_pred == 1).sum()
        self.total_acc = metrics.accuracy_score(Y_tst, Y_pred)
        self.precision = metrics.precision_score(Y_tst, Y_pred)
        self.recall = metrics.recall_score(Y_tst, Y_pred)
        self.F1 = metrics.f1_score(Y_tst, Y_pred)


    def __str__(self):
        str_results = 'Model: {}\n'.format(self.model)
        str_results += '\n=======Confusion Matrix=======\n'
        str_results += '{:20s}{:20s}{:20s}\n'.format('','classidied as 1', 'classified as 0')
        str_results += '{:20s}{:15d}{:15d}\n'.format('1 in tst set', self.true_pos, self.false_neg)
        str_results += '{:20s}{:15d}{:15d}\n\n'.format('0 in tst set', self.false_pos, self.true_neg)
        str_results += '{:50s}:{:10d}\n'.format('Total neg in tst set', self.total_neg_tst)
        str_results += '{:50s}:{:10d}\n'.format('Total pos in tst set', self.total_pos_tst)
        str_results += '{:50s}:{:10d}\n'.format('Total neg in pred', self.total_neg_pred)
        str_results += '{:50s}:{:10d}\n\n'.format('Total pos in pred', self.total_pos_pred)
        str_results += '{:50s}:{:10.3f}%\n'.format('True pos / Total pos', 100*(self.true_pos/self.total_pos_tst))
        str_results += '{:50s}:{:10.3f}%\n\n'.format('True neg / Total neg', 100*(self.true_neg/self.total_neg_tst))
        str_results += '{:50s}:{:10.3f}%\n'.format('Total accuracy score',100*self.total_acc)
        str_results += '{:50s}:{:10.3f}\n'.format('Recall = true pos / (true pos + false neg)', self.recall)
        str_results += '{:50s}:{:10.3f}\n'.format('Precision = true pos / (true pos + false pos)', self.precision)
        str_results += '{:50s}:{:10.3f}'.format('F1', self.F1)
        return str_results

if __name__ == '__main__':
    if (len(sys.argv) != 4):
        print("usage: {} <train>.csv <tsest>.csv <descriptors>.json".format(sys.argv[0]))
        exit(1)
    with open(sys.argv[3], 'r') as desc_file:
        d = json.load(desc_file)
    try:
        print("[{}] reading: train={}, test={}, desc={}".format(str(datetime.datetime.now()),
                                                       sys.argv[1], sys.argv[2], sys.argv[3]))
        print("Training parameters: {}".format(d))
        w = d["Weights"]
        t = d["DRCThreshold"]
        g = d["gamma"]
        _c = d["C"]
        X_tr, Y_tr, X_tst, Y_tst = get_data_train_tst(sys.argv[1], sys.argv[2], d['Descriptors'], t, " DRC")
    except:
        print("Cannot read dataset")
        exit(2)
    print("reading successful")
    
    weights = {0:w["0"], 1:w["1"]}
    print("[{}] start fitting".format(str(datetime.datetime.now())))
    clf = svm.SVC(C=_c, degree=3, gamma=g, class_weight=weights, cache_size=1000) 
    clf.fit(X_tr, Y_tr)
    print("[{}] fitting successfull".format(str(str(datetime.datetime.now()))))
    m = Metrics()
    m.compute_metrics(clf, X_tst, Y_tst)
    print(m)
    print("[{}] job done".format(str(datetime.datetime.now())))
