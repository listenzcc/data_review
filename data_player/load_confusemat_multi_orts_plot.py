# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def plot_scores(scores, times, axes=None):
    # plot scores,
    # left: diag values, right: across time resolution
    # Plotting layout
    if axes is None:
        fig, axes = plt.subplots(1, 2)
    # Plot Decoding over time
    ax = axes[0]
    im = ax.plot(times, np.diag(scores), label='score')
    ax.axhline(1/6, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('ACC')
    # ax.legend()
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
    return fig


def shrink_to_scores(mat_4d):
    # this function is to squeeze 1st and 2nd dim by mean values
    # it is designed to mean different ort conditions
    tmp = np.mean(mat_4d, 0)
    return np.mean(tmp, 0)


def plot_confuse_mat(confuse_mat, times):
    # this is used to plot confume_mat in a full manner
    # compare each two orts
    fig, axes = plt.subplots(6, 6)
    for j in range(6):
        for k in range(6):
            if j >= k:
                continue
            scores = shrink_to_scores(confuse_mat[j][k])
            ax = axes[j][k]
            plot_scores(scores, times, axes=[axes[j][k], axes[k][j]])
            continue
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
    return fig


mat_fname = 'confuse_mat_multi_cross_ZYF_lr_10hz'
confuse_mat = np.load('pics/%s.npy' % mat_fname)

# confuse_mat's shape is 5 x 100 x 101 x 101
# 5 cross x 100 repeats x 101 times x 101 times
confuse_mat = np.transpose(confuse_mat, [2, 3, 0, 1])

times = np.linspace(-0.2, 0.8, 101)
fig = plot_scores(shrink_to_scores(confuse_mat), times)
fig.set_figwidth(10)
fig.set_figheight(5)
fig.savefig('pics/%s_0.png' % mat_fname, dpi=300)

plt.show()
