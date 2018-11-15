# coding: utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from load_preprocess import get_epochs
sys.path.append('C:\\Users\\liste\\Documents\\Python Scripts\\clock_tools')
from simple_timer import simple_timer

# Prepare filename QYJ, ZYF
filedir = 'D:/BeidaShuju/rawdata/ZYF'


def para_setting(train=True, filedir=filedir):
    if train:
        fname_list = list(os.path.join(
            filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
            for j in range(1, 6))
        ortids = [2, 6, 9, 14, 17, 33]
        event_ids = dict(ort015=2,  ort045=6,  ort075=9,
                         ort105=14, ort135=17, ort165=33)
        tmin, t0, tmax = -0.25, 0, 1.25
    else:
        fname_list = list(os.path.join(
            filedir, 'MultiTest_%d_raw_tsss.fif' % j)
            for j in range(1, 9))
        ortids = [8, 16, 32, 64]
        event_ids = dict(ort45a=8, ort135a=16,
                         ort45b=32, ort135b=64)
        tmin, t0, tmax = -0.4, -0.2, 1.25

    return fname_list, ortids, event_ids, tmin, t0, tmax


def plot_evoked(evoked2plot, title='noname'):
    fig = evoked2plot.plot(spatial_colors=True, gfp=True, show=False,
                           time_unit='s', window_title=title)
    axes = fig.get_axes()
    for j in [0, 1]:
        ylim = axes[j].get_ylim()
        axes[j].plot([0, 0], ylim)
    return fig, title


st = simple_timer()
train = False
fname_list, ortids, event_ids, tmin, t0, tmax = para_setting(train=train)
# fname = fname_list[0]
for fname in fname_list:
    print(fname)
    epochs = get_epochs(fname=fname, event_id=event_ids,
                        tmin=tmin, t0=t0, tmax=tmax,
                        good_sensors=False)
    evoked = epochs.average()

    fig, title = plot_evoked(
        evoked, title=os.path.basename(fname[0:-4]))

    fig.savefig(os.path.join('pics', title+'.png'), dpi=600)

    st.click()

plt.close('all')
