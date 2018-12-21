# coding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.decoding import GeneralizingEstimator
from tools_loaddata import para_setting, get_epochs

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.append('C:\\Users\\liste\\Documents\\Python Scripts\\clock_tools')
from simple_timer import simple_timer

st = simple_timer()
train = True
fname_list, ortids, event_ids, tmin, t0, tmax = para_setting(train=train)

t = np.triu(np.ones([1001, 1001])/1001, 0)
ts = np.linspace(tmin, tmax, 1001)

epochs_run = []
for fname in fname_list:
    print(fname)
    epochs, raw = get_epochs(fname=fname, event_id=event_ids,
                             tmin=tmin, t0=t0, tmax=tmax,
                             freq_l=1, freq_h=5,
                             use_good_sensors=False,
                             get_envlop=False)
    epochs_run.append(epochs)

    evoked = epochs.average()
    evoked.plot_topo(show=False)
    # data = evoked.data
    # plt.figure()
    # plt.plot(ts, data.transpose())
    st.click()

X_train = np.vstack(epochs_run[j].get_data() for j in range(4))
y_train = np.vstack(epochs_run[j].events for j in range(4))[:, 2]
X_test = epochs_run[4].get_data()
y_test = epochs_run[4].events[:, 2]

n_jobs = 6
clf = make_pipeline(StandardScaler(), LogisticRegression())
time_gen = GeneralizingEstimator(clf, scoring='accuracy', n_jobs=n_jobs)
time_gen.fit(X=X_train, y=y_train)
scores = time_gen.score(X=X_test, y=y_test)

# Plotting layout
fig, axes = plt.subplots(1, 2)

# Plot Decoding over time
ax = axes[0]
im = ax.plot(epochs.times, np.diag(scores), label='score')
ax.axhline(1/6, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('ACC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')

# Plot Generalization over time
ax = axes[1]
im = ax.matshow(scores, vmin=0, vmax=1., cmap='RdBu_r', origin='lower',
                extent=epochs.times[[0, -1, 0, -1]])
ax.axhline(0., color='k')
ax.axvline(0., color='k')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Generalization across time and condition')
plt.colorbar(im, ax=ax)

plt.show()
