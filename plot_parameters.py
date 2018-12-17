# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
from load_preprocess_view import get_epochs
sys.path.append('C:\\Users\\liste\\Documents\\Python Scripts\\clock_tools')
from simple_timer import simple_timer

filedir = 'D:/BeidaShuju/rawdata/ZYF'


def para_setting(train=True, filedir=filedir):
    if train:
        fname_list = list(os.path.join(
            filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
            for j in range(1, 6))
        ortids = [2, 6, 9, 14, 17, 33]
        event_ids = dict(ort015=2,  ort045=6,  ort075=9,
                         ort105=14, ort135=17, ort165=33)
        tmin, t0, tmax = -0.2, 0, 0.8
    else:
        fname_list = list(os.path.join(
            filedir, 'MultiTest_%d_raw_tsss.fif' % j)
            for j in range(1, 9))
        ortids = [8, 16, 32, 64]
        event_ids = dict(ort45a=8, ort135a=16,
                         ort45b=32, ort135b=64)
        tmin, t0, tmax = -0.4, -0.2, 0.8
    return fname_list, ortids, event_ids, tmin, t0, tmax


def gabor(para_, t):
    A = para_[0]
    A0 = para_[1]
    t0 = para_[2]
    d = para_[3]
    w = para_[4]
    p = para_[5]

    E = np.exp(-(t-t0)**2/d/d)
    C = np.cos(w*t+p)

    return A * E * C + A0


st = simple_timer()
train = True
fname_list, ortids, event_ids, tmin, t0, tmax = para_setting(train=train)

for fname in fname_list:
    print(fname)
    basename = os.path.basename(fname[0:-4])
    epochs, raw = get_epochs(fname=fname, event_id=event_ids,
                             tmin=tmin, t0=t0, tmax=tmax,
                             use_good_sensors=False)
    evoked = epochs.average()

    map_ch_names = np.zeros(306)
    for j in range(306):
        if evoked.ch_names[j][-1] == '1':
            map_ch_names[j] = 1
    evoked_meg = evoked.copy().pick_types(meg='mag')
    evoked_meg.plot_topo(show=False)

    p_list = ['A', 'A0', 't0', 'd', 'w', 'p', 'loss']
    para2show = {}
    for p_ in p_list:
        if p_ == 'loss':
            continue
        para_path = os.path.join('matlab_encoding',
                                 'results_naive_scale',
                                 basename+'_para_%s.txt.mat' % p_)
        para_ = np.loadtxt(para_path)
        para2show[p_] = para_[map_ch_names == 1, :]
    para_path = os.path.join('matlab_encoding',
                             'results_naive_scale',
                             basename+'_loss_train.txt.mat')
    para_ = np.loadtxt(para_path)
    para2show['loss'] = para_[map_ch_names == 1, :]

    fig, axes = plt.subplots(len(p_list), 6)
    jp_ = 0
    for p_ in p_list:
        for j in range(6):
            mne.viz.plot_topomap(para2show[p_][:, j],
                                 evoked_meg.info,
                                 show=False,
                                 axes=axes[jp_, j])
            axes[jp_, j].set_title('%s, %d' % (p_, j))
        jp_ += 1

    fig.savefig(os.path.join('pics', basename+'para.png'),
                dpi=300)

    fig1, axes1 = plt.subplots(2, 3)
    fig2, axes2 = plt.subplots(2, 3)
    for o_ in range(6):
        gabord = np.zeros([102, 1001])
        t = np.linspace(-0.2, 0.8, 1001)
        for j in range(102):
            para_ = [
                para2show['A'][j, o_],
                para2show['A0'][j, o_],
                para2show['t0'][j, o_],
                para2show['d'][j, o_],
                para2show['w'][j, o_],
                para2show['p'][j, o_],
            ]
            gabord[j] = gabor(para_, t)

        a1 = axes1[o_//3][o_ % 3]
        a2 = axes2[o_//3][o_ % 3]
        evoked_meg_ = epochs[list(event_ids.keys())[
            o_]].average().pick_types(meg='mag')
        evoked_meg_.plot_topo(axes=a1, show=False)
        evoked_meg_.data = gabord
        evoked_meg_.plot_topo(axes=a2, show=False)
        a1.set_title('%d' % o_)
        a2.set_title('%d' % o_)

    fig1.savefig(os.path.join('pics', basename+'raw.png'),
                 dpi=300)
    fig2.savefig(os.path.join('pics', basename+'gabor.png'),
                 dpi=300)
    st.click()

plt.show()
