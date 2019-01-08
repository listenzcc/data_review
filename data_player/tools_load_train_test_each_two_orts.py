# coding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt

import itertools

from mne.decoding import GeneralizingEstimator
from tools_loaddata import para_setting, get_epochs

sys.path.append('C:\\Users\\liste\\Documents\\Python Scripts\\clock_tools')
from simple_timer import simple_timer


def train_test(X_train, y_train, X_test, y_test,
               clf, scoring, n_jobs):
    # train and test
    time_gen = GeneralizingEstimator(clf,
                                     scoring=scoring,
                                     n_jobs=n_jobs)
    time_gen.fit(X=X_train, y=y_train)
    return time_gen.score(X=X_test, y=y_test)


def plot_scores(scores, times, axes=None):
    # plot scores,
    # left: diag values, right: across time resolution
    # Plotting layout
    if axes is None:
        fig, axes = plt.subplots(1, 2)
    # Plot Decoding over time
    ax = axes[0]
    im = ax.plot(times, np.diag(scores), label='score')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('ACC')
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding MEG sensors over time')
    # Plot Generalization over time
    ax = axes[1]
    im = ax.matshow(scores, vmin=0, vmax=1., cmap='RdBu_r',
                    origin='lower',
                    extent=times[[0, -1, 0, -1]])
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Generalization across time and condition')
    plt.colorbar(im, ax=ax)


def load_train_test(clf, filedir, confuse_mat_path, freq_h=15,
                    use_good_sensors=False,
                    decim=10, n_jobs=12, scoring='accuracy'):

    # initial running timer
    st = simple_timer()

    # parameters setting
    fname_list, ortids, event_ids, tmin, t0, tmax = para_setting(
        train=True, filedir=filedir)

    # load raw data and epochs
    epochs_run = []
    for fname in fname_list:
        print(fname)
        epochs, raw = get_epochs(fname=fname, event_id=event_ids,
                                 tmin=tmin, t0=t0, tmax=tmax,
                                 freq_l=1, freq_h=freq_h,
                                 decim=decim,
                                 use_good_sensors=use_good_sensors,
                                 get_envlop=False)
        epochs_run.append(epochs)
        st.click()

    num_repeat = 100
    num_ort = 6
    num_cross = 5
    num_timepoint = epochs.get_data().shape[-1]
    confuse_mat = np.zeros([num_ort, num_ort,
                            num_timepoint, num_timepoint,
                            num_cross, num_repeat])
    # stack data
    X_all = np.vstack(epochs_run[j].get_data() for j in range(5))
    y_all = np.vstack(epochs_run[j].events for j in range(5))[:, 2]
    idx_list = np.unique(y_all)
    n = len(X_all)

    st.click()

    for rep_ in range(num_repeat):
        # shuffle data
        s_ = np.random.permutation(range(n))
        X_shuff = X_all.copy()[s_]
        y_shuff = y_all.copy()[s_]

        # poke data into different orts(referred as idx_)
        X_dict = {}
        y_dict = {}
        for i in range(len(idx_list)):
            idx_ = idx_list[i]
            tmp = X_shuff[y_shuff == idx_]
            X_dict[idx_] = np.vstack(np.expand_dims(
                np.mean(tmp[j*12+0:j*12+12], 0), 0) for j in range(5))
            y_dict[idx_] = np.vstack(i+1+np.zeros(len(X_dict[idx_])))

        # for each combin, seperate train and test data
        for combin_ in itertools.combinations(range(len(idx_list)), 2):
            combin = list(idx_list[j] for j in combin_)
            print(rep_, combin_, combin)
            for cross_ in range(5):
                cross_train = [0, 1, 2, 3, 4]
                cross_train.pop(cross_)
                X_train = np.vstack(X_dict[j][cross_train] for j in combin)
                y_train = np.ravel(
                    np.vstack(y_dict[j][cross_train] for j in combin))
                X_test = np.vstack(
                    np.expand_dims(X_dict[j][cross_], 0) for j in combin)
                y_test = np.ravel(np.vstack(
                    np.expand_dims(y_dict[j][cross_], 0) for j in combin))
                # train and test
                scores = train_test(X_train, y_train, X_test, y_test,
                                    clf=clf, scoring=scoring, n_jobs=n_jobs)
                confuse_mat[combin_[0], combin_[1],
                            :, :, cross_, rep_] = scores
                st.click()

    np.save(confuse_mat_path, confuse_mat)
