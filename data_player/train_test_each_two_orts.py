# coding: utf-8
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tools_load_train_test_each_two_orts import load_train_test
import os

assert(os.path.isdir('pics'))

clf = make_pipeline(StandardScaler(), LogisticRegression())
# clf = make_pipeline(StandardScaler(), LogisticRegression(
#     solver='liblinear', penalty='l1'))
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
scoring = 'accuracy'

for name in ['ZYF', 'QYJ']:
    for freq_h in [5, 10]:
        filedir = 'D:/BeidaShuju/rawdata/%s' % name
        savepath = 'pics/confuse_mat_cross_%s_lr_%dHz.npy' % (name, freq_h)
        load_train_test(clf, filedir=filedir,
                        freq_h=freq_h,
                        confuse_mat_path=savepath)
