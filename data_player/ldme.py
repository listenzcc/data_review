# coding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt

import itertools

import mne
from mne.decoding import GeneralizingEstimator
from tools_loaddata import para_setting, get_epochs

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sys.path.append('C:\\Users\\liste\\Documents\\Python Scripts\\clock_tools')
from simple_timer import simple_timer

n_jobs = 6
clf = make_pipeline(StandardScaler(), LogisticRegression())
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
scoring = 'accuracy'


def train_test(X_train, y_train, X_test, y_test,
               clf=clf, scoring=scoring, n_jobs=n_jobs):
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


# initial running timer
st = simple_timer()

# parameters setting
train = True
fname_list, ortids, event_ids, tmin, t0, tmax = para_setting(train=train)

# load raw data and epochs
epochs_run = []
for fname in fname_list:
    print(fname)
    epochs, raw = get_epochs(fname=fname, event_id=event_ids,
                             tmin=tmin, t0=t0, tmax=tmax,
                             freq_l=1, freq_h=15,
                             decim=10,
                             use_good_sensors=False,
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
        print(combin_, combin)
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
            scores = train_test(X_train, y_train, X_test, y_test)
            confuse_mat[combin_[0], combin_[1], :, :, cross_, rep_] = scores
            st.click()

np.save('pics/confuse_mat_cross.npy', confuse_mat)
# confuse_mat's shape is 6 x 6 x 5 x 100 x 101 x 101
# 6 orts x 6 orts x 5 cross x 100 repeats x 101 times x 101 times
confuse_mat = np.transpose(confuse_mat, [0, 1, 4, 5, 2, 3])


def shrink_to_scores(mat_4d):
    tmp = np.mean(mat_4d, 0)
    return np.mean(tmp, 0)


def plot_confuse_mat(confuse_mat, times):
    fig, axes = plt.subplots(6, 6)
    for j in range(6):
        for k in range(6):
            scores = shrink_to_scores(confuse_mat[j][k])
            ax = axes[j][k]
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


times = epochs.times
# plot_scores(scores, times)
plot_confuse_mat(confuse_mat, times)

ort_combine = dict()
ort_combine[30] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
ort_combine[60] = [(0, 2), (1, 3), (2, 4), (3, 5)]
ort_combine[90] = [(0, 3), (1, 4), (2, 5)]

scores_d = dict()
for dort_ in ort_combine.keys():
    confuse_mat_ = np.vstack(
        confuse_mat[e[0], e[1]] for e in ort_combine[dort_])
    scores_d[dort_] = shrink_to_scores(confuse_mat_)

fig, axes = plt.subplots(3, 2)
for dort_ in ort_combine.keys():
    plot_scores(scores_d[dort_], times, axes=axes[int(dort_/30)-1])

plt.show()
