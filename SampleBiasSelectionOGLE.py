
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import  sklearn.linear_model as linearModel
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')
import timeit
import pymc3 as pm
import matplotlib.lines as mlines
from sklearn import svm
np.random.seed(1)
from itertools import product
import sys
import json

# In[2]:
sys.path.insert(0,'./src')
import bridgeSampling as bs # it contains a method to estimate the marginal likelihood according to the Bridge Sampling approach
import utilFunctions as ut          # it has different methods to handle and plot data
import BayesianModels as bm # it has methods to build and train bayesian model (Logistic Regression and Neural Nets)
import experiments as exp
import samplebiasselection as sbs

Data, ID, Class_col, Classes = ut.Initialize(survey='OGLE')

Classes = ['rrlyr']


for i in Classes:

    if i != 'NonVar':
        Data, ID, Class_col, Classes = ut.Initialize(survey='OGLE')
        Data = Data[Data[Class_col] != 'NonVar']

        print(ID)
        print(Class_col)
        print(Classes)

        print(Data[Class_col].value_counts())

        classA = [i]
        label1 = 'ClassA'
        label2 = 'ClassB'

        Data = ut.jointClasses(classA, Data, Class_col, label1)
        print(Data[Class_col].value_counts())
        Data = ut.jointComplementClasses(classA, Classes, Data, Class_col, label2)
        print(Data[Class_col].value_counts())

        print(Data.head())
        print('Class A: obs: ', Data[Data[Class_col]==label1].count()[0])
        print('Class B: obs: ', Data[Data[Class_col]==label2].count()[0])

        print(Data.columns)
        kwargs = ut.readKwargs('experimentParameters/globalVariables.txt')
        Data = Data.dropna()
        print(Data.shape)
        Data_train, Data_test, label_train, label_test = sbs.sampleBiasSelection(Data,name=i, **kwargs)

        train = Data_train.Pred
        test = Data_test.Pred



        sbs.plot(train, test, title = classA[0]+' in OGLE')

        Data_train['label'] = np.asanyarray(label_train)
        Data_test['label'] = np.asanyarray(label_test)

        sbs.export(Data_train, Data_test, name = classA[0])
