# coding: utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
sys.path.append('..')
from load_preprocess_MVPA import get_epochs


def scale(x):
    return x * 1e14


# Prepare filename QYJ, ZYF
filedir = 'd:/BeidaShuju/rawdata/ZYF'
fname_training_list = list(os.path.join(
    filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
    for j in range(1, 6))
fname_testing_list = list(os.path.join(
    filedir, 'MultiTest_%d_raw_tsss.fif' % j)
    for j in range(1, 9))
ortids_training = [2, 6, 9, 14, 17, 33]
ortids_testing = [8, 16, 32, 64]
train = True
ortids = ortids_training
fname_list = fname_training_list

# load data
ort_name = ['ort015', 'ort045', 'ort075', 'ort105', 'ort135', 'ort165']
epochs_run = list()
data_run = list()
for fname in fname_list:
    print(fname)
    epochs = get_epochs(fname=fname, train=train,
                        envlop=False)
    epochs_run.append(epochs)
    data_ = dict()
    data_['X'] = np.vstack(epochs[ort_name[j]].get_data() for j in range(6))
    data_['y'] = np.vstack(
        j+np.ones([len(epochs[ort_name[j]].get_data()), 1]) for j in range(6))
    data_run.append(data_)

# train and test
run_all = [0, 1, 2, 3, 4]
ts = np.linspace(-0.25, 1.25, 1501)
acc_all = np.zeros([5, 1501])
for run_test in run_all:
    run_train = run_all.copy()
    run_train.pop(run_test)
    print(run_train)
    for t_ in range(200, 1201):
        X_train_ = []
        y_train_ = []
        for k in range(-5, 6):
            X_train_.append(
                np.vstack(data_run[j]['X'][:, :, t_+k] for j in run_train))
            y_train_.append(np.vstack(data_run[j]['y'] for j in run_train))
        X_train = np.vstack(X_train_)
        y_train = np.vstack(y_train_)
        X_test = data_run[run_test]['X'][:, :, t_]
        y_test = data_run[run_test]['y']
        # Train classifier
        clf = LogisticRegressionCV(multi_class='multinomial',
                                   solver='lbfgs',
                                   penalty='l2',
                                   cv=5)
        print('Training')
        clf.fit(scale(X_train), np.ravel(y_train))
        # Test classifier
        print('Testing')
        y_guess = clf.predict(scale(X_test))
        acc = 1 - np.count_nonzero(y_guess-np.ravel(y_test)) / len(y_guess)
        print(acc)
        acc_all[run_test, t_] = acc
    np.savetxt('acc_all_%d.txt' % run_test, acc_all)

fig, axes = plt.subplots(2, 1)
axes[0].plot(ts, np.mean(acc_all, axis=0))
axes[1].plot(ts, acc_all.transpose())
fig.savefig('acc_all.png', dpi=300)
plt.show()
