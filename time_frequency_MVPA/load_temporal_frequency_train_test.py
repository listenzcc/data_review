# coding: utf-8

import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import time


def para_setting(filedir, train=True):
    if train:
        fname_list = list(os.path.join(
            filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
            for j in range(1, 6))
        ortids = [2, 6, 9, 14, 17, 33]
        event_id = dict(ort015=2,  ort045=6,  ort075=9,
                        ort105=14, ort135=17, ort165=33)
        tmin, t0, tmax = -0.2, 0, 0.8
    else:
        fname_list = list(os.path.join(
            filedir, 'MultiTest_%d_raw_tsss.fif' % j)
            for j in range(1, 9))
        ortids = [8, 16, 32, 64]
        event_id = dict(ort45a=8, ort135a=16,
                        ort45b=32, ort135b=64)
        tmin, t0, tmax = -0.4, -0.2, 0.8
    reject = dict(mag=5e-12, grad=4000e-13)
    return fname_list, ortids, event_id, tmin, t0, tmax, reject


def super_trail(X, y, select_y=None):
    if select_y is None:
        select_y = np.unique(y)
    dict_X = dict()
    dict_y = dict()
    for u in select_y:
        tmp = X[y == u]
        assert(tmp.shape[0] > 55)
        np.random.shuffle(tmp)
        dict_X[u] = np.vstack(np.mean(tmp[j+0:j+10], 0, keepdims=True)
                              for j in [0, 10, 20, 30, 40, 50])
        dict_y[u] = np.array([u, u, u, u, u, u])
    newX = np.vstack(dict_X[u] for u in select_y)
    newy = np.ravel(np.vstack(dict_y[u] for u in select_y))
    return newX, newy


raise Exception('This is proofed to be failed, try others.')

# Instantiate label encoder
le = LabelEncoder()

# init clf
LR = LogisticRegression(multi_class='auto', solver='lbfgs')
clf = make_pipeline(CSP(n_components=4, reg=None, log=True,
                        norm_trace=False),
                    LR)
scoring = 'accuracy'
n_splits = 6
cv = StratifiedKFold(n_splits=n_splits, shuffle=True)

# prepare epochs
filedir = 'D:/BeidaShuju/rawdata/%s' % 'ZYF'
fname_list, ortids, event_id, tmin, t0, tmax, reject = para_setting(
    train=True, filedir=filedir)
raw_files = [mne.io.read_raw_fif(f, preload=True) for f in fname_list]
raw = mne.io.concatenate_raws(raw_files)
events = mne.find_events(raw)
sfreq = raw.info['sfreq']
picks = mne.pick_types(raw.info, meg=True,
                       stim=False, exclude='bads')
epochs = mne.Epochs(raw, event_id=event_id, events=events,
                    decim=1, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=(tmin, t0),
                    reject=reject, preload=True)
y = le.fit_transform(epochs.events[:, 2])

# prepare time and frequency
min_freq = 5.
max_freq = 80.
n_freqs = 20
freqs = np.linspace(min_freq, max_freq, n_freqs)
freq_ranges = list(zip(freqs[:-1], freqs[1:]))
tmin, tmax = tmin, tmax
n_cycles = 2.
window_spacing = (n_cycles / np.max(freqs) / 2.)
centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
n_windows = len(centered_w_times)

# init scores
tf_scores = np.zeros((n_freqs - 1, n_windows))

n_jobs = 20
for freq, (fmin, fmax) in enumerate(freq_ranges):
    w_size = n_cycles / ((fmax + fmin) / 2.)
    print(freq, fmin, fmax, w_size)
    epochs_filter = epochs.copy().filter(
        fmin, fmax, n_jobs=n_jobs, fir_design='firwin')
    for t, w_time in enumerate(centered_w_times):
        w_tmin = w_time - w_size / 2.
        w_tmax = w_time + w_size / 2.
        print('---------------------------------------------------------')
        print(freq, fmin, fmax, w_size)
        print(t, len(centered_w_times), w_tmin, w_tmax)
        X = epochs.copy().crop(w_tmin, w_tmax).get_data()
        newX, newy = super_trail(X, y)
        score = cross_val_score(estimator=clf,
                                X=newX, y=newy,
                                scoring=scoring, cv=cv,
                                n_jobs=n_jobs)
        tf_scores[freq, t] = np.mean(score)

np.save('tf_scores_%s.npy' % time.ctime(), tf_scores)
