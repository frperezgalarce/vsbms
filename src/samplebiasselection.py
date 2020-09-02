# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
import sys
import utilFunctions as ut

filterwarnings('ignore')
np.random.seed(1)
sys.path.insert(0, './src')


def samplebiasselection(data, name='rrlyrae', **kwargs):
    data = data[(data[kwargs['name_class_col']] == kwargs['class_1']) |
                (data[kwargs['name_class_col']] == kwargs['class_2'])]

    label = 1 * (data[kwargs['name_class_col']] == kwargs['class_1'])
    data_biased = data.copy()

    del data_biased[kwargs['name_class_col']]
    del data_biased[kwargs['id_col']]

    clf = RandomForestClassifier(max_depth=int(kwargs['deep_Max']), random_state=0)

    try:
        data_biased = data_biased.replace('\n', '', regex=True).replace('null', '0.0', regex=True).apply(pd.to_numeric,
                                                                                                         errors='ignore')
    except:
        print('Error in replace')

    clf.fit(data_biased, label)

    pred2 = clf.predict_proba(data_biased)
    pred = clf.predict(data_biased)
    print('Acc:', accuracy_score(pred, label))
    cm = confusion_matrix(pred, label)
    ut.plot_confusion_matrix(cm, ['all', name])

    data['pred'] = pred2[:, 0].tolist()
    data['pred2'] = pred2[:, 1].tolist()

    data['h'] = 1 - data['pred'] * data['pred'] - data['pred2'] * data['pred2']
    t = float(kwargs['T'])
    factor = t * data['h']
    data['e'] = np.exp(-factor)
    data['u'] = np.random.uniform(0, 1, data.shape[0])

    bias_selection = True
    threshold = 0.8

    if bias_selection:
        data_test = data[(data['e'] <= data['u'])]
        data_train = data[(data['e'] > data['u'])]
    else:
        data_test = data[(threshold <= data['u'])]
        data_train = data[(threshold > data['u'])]

    label_train = data_train[kwargs['name_class_col']]
    label_test = data_test[kwargs['name_class_col']]

    filename = 'Results/plots/' + name + 'OGLE.txt'
    f = open(filename, 'w+')
    f.write('TRAIN\n')
    f.write(str(label_train.value_counts()) + '\n')
    f.write('TEST\n')
    f.write(str(label_test.value_counts()) + '\n')
    f.close()
    del data_train[kwargs['name_class_col']]
    del data_test[kwargs['name_class_col']]

    print('Shape training: ', data_train.shape)
    print('Shape testing: ', data_test.shape)
    return data_train, data_test, label_train, label_test


def plot(train, test, title='RRLYRAE', survey='OGLE'):
    x_w = np.empty(train.shape)
    x_w.fill(1 / train.shape[0])
    y_w = np.empty(test.shape)
    y_w.fill(1 / test.shape[0])
    bins = np.linspace(0, 1, 10)
    plt.hist([train, test], bins, weights=[x_w, y_w], label=['training set', 'testing set'])
    plt.legend(loc='best')
    plt.title(title)
    plt.xlabel('soft predict')
    plt.ylabel('normalized frequency')
    plt.savefig('Results/plots/' + title + survey + '2.png')
    plt.clf()


def export(train, test, name='RRLYRAE', survey='OGLE'):
    train.to_csv('data/BIASEDFATS/Train2_' + survey + '_' + name + '.csv')
    test.to_csv('data/BIASEDFATS/Test2_' + survey + '_' + name + '.csv')
