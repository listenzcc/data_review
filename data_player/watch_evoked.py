# coding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt

import scipy

import mne
from mne.decoding import GeneralizingEstimator
from tools_loaddata import para_setting, get_epochs

sys.path.append('C:\\Users\\liste\\Documents\\Python Scripts\\clock_tools')
from simple_timer import simple_timer

# initial running timer
st = simple_timer()


def mds(d, dimension=2):
    (n, n) = np.shape(d)
    t = np.zeros((n, n))
    d_square = d**2
    d_sum = np.sum(d_square)
    d_sum_row = np.sum(d_square, axis=0)
    d_sum_col = np.sum(d_square, axis=1)
    for i in range(n):
        for j in range(n):
            t[i, j] = -(d_square[i, j] - d_sum_row[i] /
                        n - d_sum_col[j]/n + d_sum/(n*n))/2
    [U, S, V] = np.linalg.svd(t)
    X_original = U * np.sqrt(S)
    X = X_original[:, 0:dimension]
    return X


def cal_dist_mat(data_dict, times, range_):
    orts = list(data_dict.keys())
    mat = np.zeros([7, 7])
    select = (times > range_[0]) & (times < range_[1])
    for j in range(7):
        for k in range(7):
            # d1 = np.mean(data_dict[orts[j]][:, select], 1)
            # d2 = np.mean(data_dict[orts[k]][:, select], 1)
            d1 = np.ravel(data_dict[orts[j]][:, select])
            d2 = np.ravel(data_dict[orts[k]][:, select])
            # mat[j][k] = 1 - np.corrcoef(d1, d2)[0, 1]
            mat[j][k] = 1 - scipy.stats.spearmanr(d1, d2).correlation
            # mat[j][k] = np.linalg.norm(d1-d2)
    return mat


def plt_dmat(dmat, string, axes=None, color='black', s=100):
    if axes is None:
        fig, axes = plt.subplots(1, 1)
    md = mds(dmat)
    axes.scatter(md[:, 0], md[:, 1],
                 color=color, s=s, lw=0,
                 label='dd')
    for j in range(len(string)):
        axes.text(md[j, 0], md[j, 1],
                  string[j], fontsize=10,
                  color=color)


def idx2sub(j, col=5):
    return j // col, j % col


# parameters setting
train = True
fname_list, ortids, event_ids, tmin, t0, tmax = para_setting(train=train)

# load raw data and epochs

fig, axes = plt.subplots(2, 5)
color_list = [
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5],
    [0.5, 0.5, 0],
    [0.5, 0, 0.5]
]
epochs_run = []
for fname_idx in range(5):
    fname = fname_list[fname_idx]
    print(fname)
    epochs, raw = get_epochs(fname=fname, event_id=event_ids,
                             tmin=tmin, t0=t0, tmax=tmax,
                             freq_l=0.03, freq_h=330,
                             decim=1,
                             use_good_sensors=False,
                             get_envlop=False)
    epochs.filter(l_freq=None, h_freq=15)

    times = epochs.times
    data_dict = {}
    evoked = epochs.average()
    data_dict['all'] = evoked.data

    for ort in epochs.event_id.keys():
        print(ort)
        evoked_ = epochs[ort].average()
        data_dict[ort] = evoked_.data

    times_list = np.linspace(-0.1, 0.7, 9)
    dmat_dict = dict()
    for t in times_list:
        range_ = np.array([-0.05, 0.05]) + t/10
        dmat_dict[t] = cal_dist_mat(data_dict, times, range_)

    string = list(data_dict.keys())
    for t in range(len(times_list)):
        a, b = idx2sub(t, 5)
        axe = axes[a][b]
        plt_dmat(dmat_dict[times_list[t]],
                 string=string, axes=axe,
                 s=50, color=color_list[fname_idx])
        axe.set_title('%.2f' % times_list[t])
        epochs_run.append(epochs)

    st.click()

# mean data into 6 orts and all ort section
times = epochs.times
data_dict = {}

evoked = epochs.average()
evoked.data = np.mean(np.vstack(np.expand_dims(
    epochs_run[j].average().data, 0)
    for j in range(5)), 0)
# evoked.plot(show=False, time_unit='s', spatial_colors=True)
data_dict['all'] = evoked.data

for ort in epochs.event_id.keys():
    print(ort)
    evoked_ = epochs[ort].average().copy()
    evoked_.data = np.mean(np.vstack(np.expand_dims(
        epochs_run[j][ort].average().data, 0)
        for j in range(5)), 0)
    # evoked_.plot(show=False, time_unit='s', spatial_colors=True)
    data_dict[ort] = evoked_.data

# distance analysis for each time range
times_list = np.linspace(-0.1, 0.7, 9)
dmat_dict = dict()
for t in times_list:
    range_ = np.array([-0.05, 0.05]) + t/10
    dmat_dict[t] = cal_dist_mat(data_dict, times, range_)

string = list(data_dict.keys())
for t in range(len(times_list)):
    a, b = idx2sub(t, 5)
    axe = axes[a][b]
    plt_dmat(dmat_dict[times_list[t]],
             string=string, axes=axe)
    axe.set_title('%.2f' % times_list[t])

plt.show()
