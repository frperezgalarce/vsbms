import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as linearModel
import itertools

from warnings import filterwarnings

filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pymc3.theanof import set_tt_rng, MRG_RandomStreams
from sklearn.manifold import TSNE
from sklearn.utils import resample

set_tt_rng(MRG_RandomStreams(42))
from itertools import cycle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def initialize_data(survey='OGLE', sep_columns=' ', sep_header=' ', max_sample=5000000):
    ' surveys : OGLE, GAIA, VVV, WISE '
    path = 'FATS/'
    if survey == 'OGLE':
        print('Running OGLE')
        data = readFileFats(path + 'OGLE_FATS_12022019.csv', formatFile='.csv', sepColumns=sep_columns,
                            sepHeader=sep_header)
        ID = 'ID'
        class_col = 'Class'
        classes = data.Class.unique()

    if survey == 'GAIA':
        print('Running GAIA')
        data = readFileFats(path + 'FATS_GAIA.dat', formatFile='.dat', sepColumns=sep_columns, sepHeader=sep_header)
        ID = 'ID'
        class_col = 'Class'
        classes = data.Class.unique()

    if survey == 'MACHO':
        print('Running MACHO')
        data = readFileFats(path + 'FATS_MACHO_lukas2.dat', sepColumns=sep_columns, sepHeader=sep_header)
        ID = 'ID'
        class_col = 'Class'
        classes = data.Class.unique()

    if survey == 'VVV':
        print('Running VVV')
        data = readFileFats(path + 'FATS_VVV.dat', formatFile='.dat', sepColumns=sep_columns, sepHeader=sep_header)
        ID = 'ID'
        class_col = 'Class'
        classes = data.Class.unique()

    if survey == 'WISE':
        print('Running WISE')
        data = readFileFats(path + 'FATS_WISE.dat', formatFile='.dat', sepColumns=sep_columns, sepHeader=sep_header)
        ID = 'ID'
        class_col = 'Class'
        classes = data.Class.unique()

    samples = data.shape[0]
    if samples > max_sample:
        samples = max_sample
    print('The dataset contains:', samples, 'samples')
    data = data.sample(samples)

    return data, ID, class_col, classes


def most_important_features(data, features):
    for i in data.columns:
        if i not in features:
            del data[i]
    return data


def read_kwargs(path):
    kwargs = dict()
    with open(path) as raw_data:
        for item in raw_data:
            if ':' in item:
                key, value = item.split(':', 1)
                value = value.replace(",\n", "")
                kwargs[key] = value
            else:
                pass
    print(kwargs)
    return kwargs


def polynomial(data, p):
    phi = np.ones((data.shape[0], p * (data.shape[1])))
    col = 0
    data = (data - data.mean()) / data.std()
    for i in range(1, p + 1):
        for k in range(0, data.shape[1]):
            for j in range(0, data.shape[0]):
                try:
                    phi[j, col] = np.power(data.iloc[j, k], i)
                except:
                    phi[j, col] = np.power(data[j, k], i)
            col = col + 1
    ret = pd.DataFrame(phi, columns=['col' + str(i) for i in range(p * data.shape[1])]).round(3)
    return ret


def preprocess(data, delete_nonvariation=False, delete_correlation=False):
    if delete_nonvariation:
        del data['Freq2_harmonics_rel_phase_0']
        del data['Freq3_harmonics_rel_phase_0']
        del data['Freq1_harmonics_rel_phase_0']

    if delete_correlation:
        del data['Meanvariance']
        del data['Psi_CS']
        del data['Q31']
        del data['Std']
        del data['FluxPercentileRatioMid35']
        del data['FluxPercentileRatioMid20']
        del data['FluxPercentileRatioMid50']
        del data['Freq3_harmonics_amplitude_0']
        del data['PercentDifferenceFluxPercentile']
    return data


def logistic_function_(z):
    pred = 1. / (1. + np.exp(-z))
    return pred


def plot_q(logml0):
    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    for dist in ['q11', 'q12', 'q21', 'q22']:
        sns.distplot(logml0[dist], ax=ax, label=dist, bins=20)
    plt.legend();
    plt.show()


def surface_plot(X, Y, Z, **kwargs):
    xlabel, ylabel, zlabel, title = kwargs.get('xlabel', ""), kwargs.get('ylabel', ""),\
                                    kwargs.get('zlabel', ""), kwargs.get('title', "")
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    bar = kwargs['bar']
    if bar == True:
        ax.bar(Y, -Z, zs=X, zdir='x')
    else:
        ml_max = Z.max()
        hidden_max = Y.iloc[Z.idxmax()]
        component_max = X.iloc[Z.idxmax()]
        Y.drop(Z.idxmax())
        X.drop(Z.idxmax())
        Z.drop(Z.idxmax())
        ax.scatter(component_max, hidden_max, ml_max, color='red', linewidth=5, marker='o')
        ax.scatter(X, Y, Z, linewidth=3, marker='o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()
    plt.close()


def get_z(data, trace, burn_in=1000):
    r = np.mean(trace.get_values('Intercept', burn=burn_in, combine=False)[0])
    try:
        del data['label']
    except:
        print('getting z')
    for i in range(data.shape[1]):
        it = data.columns[i]
        values = np.mean(trace.get_values(it, burn=burn_in, combine=False)[0])
        r = np.round(r + np.outer(data.loc[:, it], values), 5)
    return r


def plot_TSNE(data, labels, perplexity_=100, n_iter_=3000, verbose_=0):
    n_sne = data.shape[0]
    np_scaled = preprocessing.normalize(data[0:n_sne])
    df_normalized = pd.DataFrame(np_scaled)
    tsne = TSNE(n_components=2, verbose=verbose_, perplexity=perplexity_, n_iter=n_iter_)
    tsne_results = tsne.fit_transform(df_normalized)
    df_tsne_1 = pd.DataFrame()
    df_tsne_2 = pd.DataFrame()
    df_tsne_1['class'] = labels[0:n_sne]
    df_tsne_2['x-tsne'] = tsne_results[:, 0]
    df_tsne_2['y-tsne'] = tsne_results[:, 1]
    df_tsne_1.reset_index(drop=True, inplace=True)
    df_tsne_2.reset_index(drop=True, inplace=True)
    df_tsne = pd.concat([df_tsne_1, df_tsne_2], axis=1, ignore_index=True)
    cycol = cycle('bgrcmk')
    for i in labels.unique():
        data_mew = df_tsne[df_tsne[0] == i]
        plt.scatter(data_mew[1], data_mew[2], color=next(cycol), label=i)
    plt.legend()
    plt.show()


def down_sampling(df):
    df_a = df[df.label == 'class_a']
    df_b = df[df.label == 'ClassB']

    if df_a.shape[0] > df_b.shape[0]:
        df_majority = df_a
        df_minority = df_b
    else:
        df_majority = df_b
        df_minority = df_a

    df_majority_downsampled = resample(df_majority, replace=False,
                                       n_samples=df_minority.shape[0], random_state=123)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    return df_downsampled


def dim_reduction(data, label, components, type_net='bernoulli'):
    pca = PCA(n_components=components)
    pca.fit(data)
    data = pca.transform(data)
    data = pd.DataFrame(data)

    if type_net == 'categorical':
        ym = label.as_matrix()
    if components == 0:
        fig, ax = plt.subplots()
        ax.scatter(data[label == 0, 0], data[label == 0, 1], label='Class 0')
        ax.scatter(data[label == 1, 0], data[label == 1, 1], color='r', label='Class 1')
        ax.set(xlabel='X', ylabel='Y', title='Toy binary classification data set');
        plt.show()
    return data, label


def joint_classes(class_a, data, class_col, label1):
    print('classes: ', class_a)
    data[class_col] = data[class_col].replace(class_a, label1)
    return data


def delete_class(deleteclass, data, class_col):
    data = data[data[class_col] != deleteclass]
    return data


def jointComplementClasses(classA, Classes, Data, Class_col, label2):
    complement = []
    for c in Classes:
        if c not in classA:
            complement.append(c)
    Data = joint_classes(complement, Data, Class_col, label2)
    return Data


def Define_TrainSet(Data, name_class_col='Class', id_col='ID', plot=False,
                    test=0.2, biasedSplit=False, Features=10, classRef='RRLYR',
                    class_2='RRLYR', alpha=0.1, DropEasy=False,
                    oneToOne=True, PCA=True, **kwargs):
    Data[name_class_col] = pd.Categorical(Data[name_class_col])
    classes_Data = Data[name_class_col].unique()
    print('Running Define TrainSet')
    print(Data.shape)

    if (plot == True):
        ym_ = Data.class_name
        plt.figure(figsize=(8, 8))
        plt.hist(ym_)
        plt.title("Classes complete dataset")
        plt.show()

    if biasedSplit == False:
        print('Test size: ', int(Data.shape[0] * test))
        Data_train, Data_test = train_test_split(Data, test_size=test, random_state=42)
        label_train = Data_train[name_class_col]
        label_test = Data_test[name_class_col]
        del Data_train[id_col]
        del Data_test[id_col]
        del Data_train[name_class_col]
        del Data_test[name_class_col]
        print('Shape training: ', Data_train.shape)
        print('Shape testing: ', Data_test.shape)
        Data_train = Data_train.replace('\n', '', regex=True).replace('null', '0.0', regex=True).apply(pd.to_numeric,
                                                                                                       errors='ignore')
        Data_test = Data_test.replace('\n', '', regex=True).replace('null', '0.0', regex=True).apply(pd.to_numeric,
                                                                                                     errors='ignore')
        return Data_train, Data_test, label_train, label_test
    else:
        trainFile = kwargs['trainFile']
        testFile = kwargs['testFile']
        Data_train = pd.read_csv(trainFile)
        Data_test = pd.read_csv(testFile)
        label_train = Data_train[name_class_col]
        label_test = Data_test[name_class_col]

        del Data_train[id_col]
        del Data_test[id_col]
        del Data_train[name_class_col]
        del Data_test[name_class_col]

    return Data_train, Data_test, label_train, label_test


def readFileFats(file, formatFile='.dat', sepColumns=',', sepHeader='\t'):
    path = 'data/'
    file = path + file
    if (formatFile == '.dat'):
        file = open(file)
        lst = []
        columns = []
        count = 0
        bad = 0
        for line in file:
            if count > 0:
                # print(line.split(sepColumns))
                if (len(line.split(sepColumns)) > len(columns[0])):
                    bad = bad + 1
                else:
                    lst.append(line.split(sepColumns))
            else:
                columns.append(line.split(sepHeader))
                # print(line.split(sepHeader))
            count = count + 1
        print(bad, 'lines fail when were reading')
        Data = pd.DataFrame(lst, columns=columns[0])
        return Data
    else:
        if (formatFile == '.csv'):
            print('Here')
            Data = pd.read_csv(file)
            return Data
        else:
            print('Problems with file format.')


def plot_confusion_matrix(cm, classes, type='train',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Results/plots/cm_' + classes[1] + '_.png')
    plt.show()
    plt.clf()


def KFoldLogReg(dataTrain, labelTrain, dataTest, labelTest, n_split_test, n_split_train):
    acc_kfold = []
    skf = StratifiedKFold(n_splits=int(n_split_train))
    clf = linearModel.LogisticRegression(C=1.0)
    skf.get_n_splits(dataTrain, labelTrain)

    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(dataTrain, labelTrain):
        X_train, X_test = dataTrain.iloc[train_index, :], dataTrain.iloc[test_index, :]
        y_train, y_test = labelTrain.iloc[train_index], labelTrain.iloc[test_index]
        clf.fit(X_train, y_train)
        prediction_freq = clf.predict(X_test)
        acc_kfold.append(accuracy_score(y_test, prediction_freq, normalize=True))
        acc_kfold_Test = []

    skf = StratifiedKFold(n_splits=int(n_split_test))
    clf.fit(dataTrain, labelTrain)

    skf.get_n_splits(dataTest, labelTest)
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(dataTest, labelTest):
        X_train, X_test = dataTest.iloc[train_index, :], dataTest.iloc[test_index, :]
        y_train, y_test = labelTest.iloc[train_index], labelTest.iloc[test_index]
        prediction_freq = clf.predict(X_test)
        acc_kfold_Test.append(accuracy_score(y_test, prediction_freq, normalize=True))
    return acc_kfold, acc_kfold_Test


def KFold(dataTrain, labelTrain, dataTest, labelTest, n_split_test, n_split_train, clf):
    print('Training...')
    acc_kfold = []
    acc_kfold_Test = []
    f1_kfold_Train = []
    f1_kfold_Test = []

    skf = StratifiedKFold(n_splits=int(n_split_train))
    skf.get_n_splits(dataTrain, labelTrain)

    for train_index, test_index in skf.split(dataTrain, labelTrain):
        X_train, X_test = dataTrain.iloc[train_index, :], dataTrain.iloc[test_index, :]
        y_train, y_test = labelTrain.iloc[train_index], labelTrain.iloc[test_index]
        clf.fit(X_train, y_train)
        prediction_freq = clf.predict(X_test)
        acc_kfold.append(accuracy_score(y_test, prediction_freq, normalize=True))
        print('Accuracy: ', accuracy_score(y_test, prediction_freq, normalize=True))
        print('F1-score: ', f1_score(y_test, prediction_freq, pos_label='class_a'))
        f1_kfold_Train.append(f1_score(y_test, prediction_freq, pos_label='class_a'))

    clf.fit(dataTrain, labelTrain)
    print('Testing...')

    skf = StratifiedKFold(n_splits=int(n_split_test))
    skf.get_n_splits(dataTest, labelTest)

    for train_index, test_index in skf.split(dataTest, labelTest):
        X_train, X_test = dataTest.iloc[train_index, :], dataTest.iloc[test_index, :]
        y_train, y_test = labelTest.iloc[train_index], labelTest.iloc[test_index]
        prediction_freq = clf.predict(X_test)
        print('Accuracy: ', accuracy_score(y_test, prediction_freq, normalize=True))
        print('F1-score: ', f1_score(y_test, prediction_freq, pos_label='class_a'))
        acc_kfold_Test.append(accuracy_score(y_test, prediction_freq, normalize=True))
        f1_kfold_Test.append(f1_score(y_test, prediction_freq, pos_label='class_a'))
    return acc_kfold, acc_kfold_Test, f1_kfold_Train, f1_kfold_Test


def comparativePlotAcc(Test, Train, classStar='rrlyr', clf="Random Forest", normalized=True, num_bin=10, plType='acc',
                       binslim=True):
    if plType == 'acc':
        bin_lims = np.linspace(np.min(Test), 1, num_bin + 1)
        bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
        bin_widths = bin_lims[1:] - bin_lims[:-1]
        if binslim == True:
            hist1, _ = np.histogram(np.asarray(Test), bins=bin_lims)
            hist2, _ = np.histogram(np.asarray(Train), bins=bin_lims)
        else:
            hist1, _ = np.histogram(np.asarray(Test))
            hist2, _ = np.histogram(np.asarray(Train))
        if normalized == True:
            hist1 = hist1 / np.max(hist1)
            hist2 = hist2 / np.max(hist2)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax2.bar(bin_centers, hist1, width=bin_widths, align='center', color='r', alpha=0.4)
        ax1.bar(bin_centers, hist2, width=bin_widths, align='center', alpha=0.4)
        ax2.set_title('Accuracy in Testing')
        ax1.set_title('Accuracy in Training')
        plt.savefig('Results/plots/' + classStar + '_Experiment1_ACC_' + clf + '_' + '.png')
        plt.clf()
    if plType == 'f1':
        bin_lims = np.linspace(np.min(Test), 1, num_bin + 1)
        bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
        bin_widths = bin_lims[1:] - bin_lims[:-1]
        ##computing the histograms
        if binslim == True:
            hist1, _ = np.histogram(np.asarray(Test), bins=bin_lims)
            hist2, _ = np.histogram(np.asarray(Train), bins=bin_lims)
        else:
            hist1, _ = np.histogram(np.asarray(Test))
            hist2, _ = np.histogram(np.asarray(Train))  ##normalizing
        if normalized == True:
            hist1 = hist1 / np.max(hist1)
            hist2 = hist2 / np.max(hist2)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax2.bar(bin_centers, hist1, width=bin_widths, align='center', color='r', alpha=0.4)
        ax1.bar(bin_centers, hist2, width=bin_widths, align='center', alpha=0.4)
        ax2.set_title('F1_score in Testing')
        ax1.set_title('F1_score in Training')
        plt.savefig('Results/plots/' + classStar + '_Experiment1_F1_' + clf + '_' + '.png')
        plt.clf()


def DataStructure(label, Data, components=10, typeNet='categorical', variable=0):
    'This method defines '
    if (typeNet == 'bernoulli'):
        ym = pd.get_dummies(label)[variable]
        Xm = preprocess(Data)
        Xm, ym = dim_reduction(Xm, ym, components, type_net=typeNet)
    else:
        ym = np.asanyarray(label)
        Xm = preprocess(Data)
        Xm, ym = dim_reduction(Xm, ym, components)

    return Xm, ym
