
# coding: utf-8

# In[ ]:

#get_ipython().magic('matplotlib inline')
import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from warnings import filterwarnings
import  sklearn.linear_model as linearModel
filterwarnings('ignore')
#sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from pymc3.theanof import set_tt_rng, MRG_RandomStreams
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.utils import resample
import itertools
set_tt_rng(MRG_RandomStreams(42))
from pymc3.model import modelcontext
from scipy import dot
from scipy import special
from scipy.linalg import cholesky as chol
from itertools import cycle
import scipy.stats as st
import warnings
from tempfile import mkdtemp
from pymc3.step_methods import smc
#import seaborn
from sklearn import preprocessing
from pymc3.variational.callbacks import CheckParametersConvergence
import timeit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import sys

def Initialize(survey = 'OGLE', sepColumns_=' ', sepHeader_= ' ', maxSample = 5000000):
    ' surveys : OGLE, GAIA, VVV, WISE '

    path = 'FATS/'
    if survey == 'OGLE':
        print('Running OGLE')
        #Data = readFileFats(path+'FATS_OGLE.dat', formatFile ='.dat', sepColumns= sepColumns_, sepHeader= sepHeader_)
        Data = readFileFats(path+'OGLE_FATS_12022019.csv', formatFile ='.csv', sepColumns= sepColumns_, sepHeader= sepHeader_)
        print(Data.head())
        ID = 'ID'
        Class_col = 'Class'
        Classes = Data.Class.unique()

    if survey == 'GAIA':
        print('Running GAIA')
        Data = readFileFats(path+'FATS_GAIA.dat', formatFile ='.dat', sepColumns= sepColumns_, sepHeader= sepHeader_)
        print(Data.head())
        ID = 'ID'
        Class_col = 'Class'
        Classes = Data.Class.unique()

    if survey == 'MACHO':
        print('Running MACHO')
        Data = readFileFats(path+'FATS_MACHO_lukas2.dat', sepColumns= sepColumns_, sepHeader= sepHeader_)
        #Data = readFileFats('FATS_OGLE.dat', formatFile ='.dat', sepColumns=',', sepHeader= ' ')
        ID = 'ID'
        Class_col = 'Class'
        Classes = Data.Class.unique()

    if survey == 'VVV':
        print('Running VVV')
        Data = readFileFats(path+'FATS_VVV.dat', formatFile ='.dat', sepColumns= sepColumns_, sepHeader= sepHeader_)
        ID = 'ID'
        Class_col = 'Class'
        Classes = Data.Class.unique()

    if survey == 'WISE':
        print('Running WISE')
        Data = readFileFats(path+'FATS_WISE.dat', formatFile ='.dat', sepColumns= sepColumns_, sepHeader= sepHeader_)
        ID = 'ID'
        Class_col = 'Class'
        Classes = Data.Class.unique()

    samples = Data.shape[0]
    if samples > maxSample:
        samples = maxSample
    print('The dataset contains:', samples, 'samples')
    Data = Data.sample(samples)

    return Data, ID, Class_col, Classes

def MostImportanFeature(Data, Features):
    for i in Data.columns:
        if i not in Features:
            del Data[i]
    return Data

def readKwargs(path):
    kwargs = dict()
    with open(path) as raw_data:
        for item in raw_data:
            if ':' in item:
                key,value = item.split(':', 1)
                value = value.replace(",\n","")
                kwargs[key]=value
            else:
                pass
    print(kwargs)
    return kwargs

def kernelGaussiano(X, mean, sigma=1):
    phi = np.ones((X.shape[0],mean.shape[0]*X.shape[1]))
    col = 0
    X = (X-X.mean())/X.std()
    for i in range(0, mean.shape[0]):
        for k in range(0, X.shape[1]):
            for j in range(0, X.shape[0]):
                phi[j,col] = norm.pdf(X.iloc[j,k], mean[i-1], sigma)
            col = col + 1
    return pd.DataFrame(phi)

def kernelPolinomial(X, p):
    phi = np.ones((X.shape[0],p*(X.shape[1])))
    col = 0
    X = (X-X.mean())/X.std()
    for i in range(1, p+1):
        for k in range(0, X.shape[1]):
            for j in range(0, X.shape[0]):
                #print('power', str(i), 'row: ',str(j),' column: ', str(i*k +i), 'data: ', X.iloc[j,k], 'result: ', np.power(X.iloc[j,k],i))
                try:
                    phi[j,col] = np.power(X.iloc[j,k],i)
                    #print(np.power(X.iloc[j,k],i))
                except:
                    phi[j,col] = np.power(X[j,k],i)
                    #print(np.power(X[j,k],i))
                #print(phi[j ,col])
            col = col + 1
    ret = pd.DataFrame(phi, columns = ['col'+str(i) for i in range(p*X.shape[1])]).round(3)
    #print(ret.head())
    return ret

def preprocess(Data, delete_noVariation = False, delete_correlation = False):
    #del Data['class_name']
    #del Data['ogle_id']
    Xm = Data.copy()
    #delete_noVariation = True
    #delete_correlation = True
    #delete_outlier = False

    if(delete_noVariation == True):
        del Xm['Freq2_harmonics_rel_phase_0']
        del Xm['Freq3_harmonics_rel_phase_0']
        del Xm['Freq1_harmonics_rel_phase_0']

    if(delete_correlation == True):
        del Xm['Meanvariance']
        del Xm['Psi_CS']
        del Xm['Q31']
        del Xm['Std']
        del Xm['FluxPercentileRatioMid35']
        del Xm['FluxPercentileRatioMid20']
        del Xm['FluxPercentileRatioMid50']
        del Xm['Freq3_harmonics_amplitude_0']
        del Xm['PercentDifferenceFluxPercentile']

    Xm.describe()
    return Xm

def logistic_function_(z):
    pred = 1./(1.+np.exp(-z))
    #print(pred)
    return pred

def surface_plot(X,Y,Z,**kwargs):
    """ WRITE DOCUMENTATION """
    xlabel, ylabel, zlabel, title = kwargs.get('xlabel',""), kwargs.get('ylabel',""), kwargs.get('zlabel',""), kwargs.get('title',"")
    fig = plt.figure(figsize=(12,8))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    #X, Y = np.meshgrid(X, Y)
    bar = kwargs['bar']
    if bar == True:
        ax.bar(Y, -Z, zs=X, zdir='x')
    else:
        mlMax = Z.max()
        hiddenMax = Y.iloc[Z.idxmax()]
        componentMax = X.iloc[Z.idxmax()]
        Y.drop(Z.idxmax())
        X.drop(Z.idxmax())
        Z.drop(Z.idxmax())
        ax.scatter(componentMax,hiddenMax, mlMax, color='red',linewidth=5, marker='o')
        ax.scatter(X,Y,Z,linewidth=3, marker='o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()
    plt.close()

def get_z(data, trace, model, burn_in = 1000):

    #ppc = pm.sample_ppc(trace=trace, samples = 100,size=100, model = model)
    r =  np.mean(trace.get_values('Intercept', burn = burn_in, combine = False)[0])
    #r = np.mean(np.asarray(ppc['Intercept']))
    try:
        del data['label']
    except:
        print('getting z')
    for i in range(data.shape[1]):
        it = data.columns[i]
        values = np.mean(trace.get_values(it, burn = burn_in, combine = False)[0])
        #values = np.mean(np.asarray(ppc[it]))
        r = np.round(r + np.outer(data.loc[:,it], values),5)
    return r

def plot_TSNE(Data, labels,  perplexity_=100, n_iter_=3000, verbose_ = 0):
    n_sne = Data.shape[0]
    np_scaled = preprocessing.normalize(Data[0:n_sne])
    df_normalized = pd.DataFrame(np_scaled)

    tsne = TSNE(n_components=2, verbose= verbose_, perplexity=perplexity_, n_iter=n_iter_)
    tsne_results = tsne.fit_transform(df_normalized)

    df_tsne_1 = pd.DataFrame()
    df_tsne_2 = pd.DataFrame()
    df_tsne_1['class'] = labels[0:n_sne]
    df_tsne_2['x-tsne'] = tsne_results[:,0]
    df_tsne_2['y-tsne'] = tsne_results[:,1]

    df_tsne_1.reset_index(drop=True, inplace=True)
    df_tsne_2.reset_index(drop=True, inplace=True)
    df_tsne = pd.concat([df_tsne_1, df_tsne_2], axis=1, ignore_index=True)

    cycol = cycle('bgrcmk')
    for i in labels.unique():
        data = df_tsne[df_tsne[0]==i]
        plt.scatter(data[1], data[2], color = next(cycol), label = i)
    plt.legend()
    plt.show()

def downSampling(df):

    df_a = df[df.label=='ClassA']
    df_b = df[df.label=='ClassB']

    if df_a.shape[0] > df_b.shape[0]:
        df_majority = df_a
        df_minority = df_b
    else:
        df_majority = df_b
        df_minority = df_a

    df_majority_downsampled = resample(df_majority, replace = False,
                                        n_samples = df_minority.shape[0], random_state = 123)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    return df_downsampled

def DimReduction(Xm, ym, components, label=1, typeNet = 'bernoulli'):

    pca = PCA(n_components=components)
    pca.fit(Xm)

    Xm = pca.transform(Xm)
    Xm = pd.DataFrame(Xm)

    if(typeNet == 'categorical'):
        ym = ym.as_matrix()
    if components == 0:
        fig, ax = plt.subplots()
        ax.scatter(Xm[ym==0, 0], Xm[ym==0, 1], label='Class 0')
        ax.scatter(Xm[ym==1, 0], Xm[ym==1, 1], color='r', label='Class 1')
        #sns.despine();
	#ax.legend()
        ax.set(xlabel='X', ylabel='Y', title='Toy binary classification data set');
        plt.shoow()
    return Xm, ym

def jointClasses(ClassA, Data, Class_col, label1):
    print('classes: ',ClassA)
    Data[Class_col] = Data[Class_col].replace(ClassA, label1)
    return Data

def deleteClass(deleteClass, Data, Class_col):
    Data = Data[Data[Class_col] != deleteClass]
    return Data

def jointComplementClasses(classA, Classes, Data, Class_col, label2):
    complement = []
    for c in Classes:
        if c not in classA:
             complement.append(c)
    Data = jointClasses(complement, Data, Class_col, label2)
    return Data

def Define_TrainSet(Data, name_class_col = 'Class', id_col = 'ID',plot = False,
                    test = 0.2, biasedSplit = False, Features = 10, classRef = 'RRLYR',
                    class_2= 'RRLYR', alpha = 0.1, DropEasy = False,
                    oneToOne = True, PCA = True, **kwargs):

    Data[name_class_col] = pd.Categorical(Data[name_class_col])
    classes_Data = Data[name_class_col].unique()
    #Data['class_name'] = Data.class_name.cat.codes
    print('Running Define TrainSet')
    print(Data.shape)

    if(plot == True):
        ym_ = Data.class_name
        plt.figure(figsize=(8,8))
        plt.hist(ym_)
        plt.title("Classes complete dataset")
        plt.show()

    if biasedSplit == False:
        print('Test size: ', int(Data.shape[0]*test))
        Data_train, Data_test = train_test_split(Data, test_size=test, random_state=42)
        label_train = Data_train[name_class_col]
        label_test = Data_test[name_class_col]
        del Data_train[id_col]
        del Data_test[id_col]
        del Data_train[name_class_col]
        del Data_test[name_class_col]
        print('Shape training: ', Data_train.shape)
        print('Shape testing: ', Data_test.shape)
        Data_train = Data_train.replace('\n', '', regex = True).replace('null', '0.0', regex = True).apply(pd.to_numeric, errors ='ignore')
        Data_test = Data_test.replace('\n', '', regex = True).replace('null', '0.0', regex = True).apply(pd.to_numeric, errors ='ignore')
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

def readFileFats(file, formatFile='.dat', sepColumns=',', sepHeader = '\t'):
    path = 'data/'
    file = path + file
    if(formatFile == '.dat'):
        file = open(file)
        lst = []
        columns = []
        count = 0
        bad = 0
        for line in file:
            if count > 0:
                #print(line.split(sepColumns))
                if(len(line.split(sepColumns))>len(columns[0])):
                    bad = bad + 1
                else:
                    lst.append(line.split(sepColumns))
            else:
                columns.append(line.split(sepHeader))
                #print(line.split(sepHeader))
            count = count + 1
        print(bad, 'lines fail when were reading')
        Data = pd.DataFrame(lst, columns= columns[0])
        return Data
    else:
        if(formatFile == '.csv'):
            print('Here')
            Data = pd.read_csv(file)
            return Data
        else:
            print('Problems with file format.')

def plot_confusion_matrix(cm, classes,type='train',
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
    plt.savefig('Results/plots/cm_'+classes[1]+'_.png')
    plt.show()
    plt.clf()

def KFoldLogReg(dataTrain, labelTrain, dataTest, labelTest, n_split_test,n_split_train):
    acc_kfold = []
    skf = StratifiedKFold(n_splits=int(n_split_train))
    clf = linearModel.LogisticRegression(C=1.0)
    skf.get_n_splits(dataTrain, labelTrain)

    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(dataTrain, labelTrain):
        X_train, X_test = dataTrain.iloc[train_index,:], dataTrain.iloc[test_index,:]
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
        X_train, X_test = dataTest.iloc[train_index,:], dataTest.iloc[test_index,:]
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
        X_train, X_test = dataTrain.iloc[train_index,:], dataTrain.iloc[test_index,:]
        y_train, y_test = labelTrain.iloc[train_index], labelTrain.iloc[test_index]
        #print('Train shape: ', X_train.shape)
        #print('Test shape: ', X_test.shape)
        clf.fit(X_train, y_train)
        prediction_freq = clf.predict(X_test)
        acc_kfold.append(accuracy_score(y_test, prediction_freq, normalize=True))
        print('Accuracy: ', accuracy_score(y_test, prediction_freq, normalize=True))
        print('F1-score: ', f1_score(y_test, prediction_freq, pos_label = 'ClassA'))
        f1_kfold_Train.append(f1_score(y_test, prediction_freq, pos_label = 'ClassA'))

    clf.fit(dataTrain, labelTrain)
    print('Testing...')

    skf = StratifiedKFold(n_splits=int(n_split_test))
    skf.get_n_splits(dataTest, labelTest)

    for train_index, test_index in skf.split(dataTest, labelTest):
        X_train, X_test = dataTest.iloc[train_index,:], dataTest.iloc[test_index,:]
        y_train, y_test = labelTest.iloc[train_index], labelTest.iloc[test_index]
        #print('Train shape: ', X_train.shape)
        #print('Test shape: ', X_test.shape)
        prediction_freq = clf.predict(X_test)
        print('Accuracy: ', accuracy_score(y_test, prediction_freq, normalize=True))
        print('F1-score: ', f1_score(y_test, prediction_freq, pos_label = 'ClassA'))
        acc_kfold_Test.append(accuracy_score(y_test, prediction_freq, normalize=True))
        f1_kfold_Test.append(f1_score(y_test, prediction_freq, pos_label = 'ClassA'))
    return acc_kfold, acc_kfold_Test, f1_kfold_Train, f1_kfold_Test

def comparativePlotAcc(Test, Train, classStar = 'rrlyr', clf = "Random Forest",normalized = True,  num_bin = 10, plType = 'acc', binslim = True):
    if plType == 'acc':
        bin_lims = np.linspace(np.min(Test),1,num_bin+1)
        bin_centers = 0.5*(bin_lims[:-1]+bin_lims[1:])
        bin_widths = bin_lims[1:]-bin_lims[:-1]
        ##computing the histograms
        if binslim == True:
            hist1, _ = np.histogram(np.asarray(Test), bins=bin_lims)
            hist2, _ = np.histogram(np.asarray(Train), bins=bin_lims)
        else:
            hist1, _ = np.histogram(np.asarray(Test))
            hist2, _ = np.histogram(np.asarray(Train))

        ##normalizing
        if normalized == True:
            hist1 = hist1/np.max(hist1)
            hist2 = hist2/np.max(hist2)
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)
        ax2.bar(bin_centers, hist1, width = bin_widths, align = 'center', color = 'r', alpha = 0.4)
        ax1.bar(bin_centers, hist2, width = bin_widths, align = 'center', alpha = 0.4)
        ax2.set_title('Accuracy in Testing')
        ax1.set_title('Accuracy in Training')
        plt.savefig('Results/plots/'+classStar+'_Experiment1_ACC_'+clf+'_'+'.png')
        plt.clf()
    if plType == 'f1':
        bin_lims = np.linspace(np.min(Test),1,num_bin+1)
        bin_centers = 0.5*(bin_lims[:-1]+bin_lims[1:])
        bin_widths = bin_lims[1:]-bin_lims[:-1]
        ##computing the histograms
        if binslim == True:
            hist1, _ = np.histogram(np.asarray(Test), bins=bin_lims)
            hist2, _ = np.histogram(np.asarray(Train), bins=bin_lims)
        else:
            hist1, _ = np.histogram(np.asarray(Test))
            hist2, _ = np.histogram(np.asarray(Train))        ##normalizing
        if normalized == True:
            hist1 = hist1/np.max(hist1)
            hist2 = hist2/np.max(hist2)
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)
        ax2.bar(bin_centers, hist1, width = bin_widths, align = 'center', color = 'r', alpha = 0.4)
        ax1.bar(bin_centers, hist2, width = bin_widths, align = 'center', alpha = 0.4)
        ax2.set_title('F1_score in Testing')
        ax1.set_title('F1_score in Training')
        plt.savefig('Results/plots/'+classStar+'_Experiment1_F1_'+clf+'_'+'.png')
        plt.clf()

def DataStructure(label, Data, components = 10, typeNet = 'categorical', variable=0):
    'This method defines '
    if(typeNet == 'bernoulli'):
        ym = pd.get_dummies(label)[variable]
        Xm = preprocess(Data)
        Xm, ym = DimReduction(Xm, ym, components, typeNet = typeNet)
    else:
        ym = np.asanyarray(label)
        Xm = preprocess(Data)
        Xm, ym = DimReduction(Xm, ym, components)

    return Xm, ym
