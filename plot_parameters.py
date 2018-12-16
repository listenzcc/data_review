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

    p_list = ['A', 'A0', 't0', 'd', 'w', 'p']
    para2show = {}
    for p_ in p_list:
        para_path = os.path.join('matlab_encoding',
                                 'results_naive_scale',
                                 basename+'_para_%s.txt.mat' % p_)
        para_ = np.loadtxt(para_path)
        para2show[p_] = para_[map_ch_names == 1, :]

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
    st.click()

plt.show()
