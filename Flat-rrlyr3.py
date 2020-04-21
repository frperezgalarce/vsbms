
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import  sklearn.linear_model as linearModel
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from warnings import filterwarnings
import timeit
import pymc3 as pm
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn import preprocessing

sys.path.insert(0,'./src')
import bridgeSampling as bs # it contains a method to estimate the marginal likelihood according to the Bridge Sampling approach
import utilFunctions as ut          # it has different methods to handle and plot data
import BayesianModels as bm # it has methods to build and train bayesian model (Logistic Regression and Neural Nets)
import experiments as exp
import samplebiasselection as sbs

fileTrain = 'data/BIASEDFATS/Train_rrlyr-3.csv'
fileTest =  'data/BIASEDFATS/Test_rrlyr-3.csv'

dataTrain = pd.read_csv(fileTrain)
dataTest = pd.read_csv(fileTest)
plot = False
Flat = False
res = []
PCA_ = True
informative = False
Flat = False
for k in [1,2,3]:
    for Components in [2,4,6,8,10,12]:
        for size in[500, 1000, 10000, 20000]:
            print(size)
            dataTrain = pd.read_csv(fileTrain)
            dataTest = pd.read_csv(fileTest)
            dataTrain = ut.downSampling(dataTrain)
            try:
                dataTrain = dataTrain.sample(size, random_state=0)
            except:
                print('sample bigger than data size')

            yTrain = 1*(dataTrain['label'] == 'ClassA')

            del dataTrain['label']
            yTest = 1*(dataTest['label'] == 'ClassA')
            del dataTest['label']

            try:
                dataTrain = dataTrain.loc[:, ~dataTrain.columns.str.contains('^Unnamed')]
                dataTrain =  dataTrain.drop(['Pred', 'Pred2', 'h', 'e', 'u','ID'], axis = 1)
                dataTrain = dataTrain.loc[:, dataTrain.var()!=0.0]


                dataTest = dataTest.loc[:, ~dataTest.columns.str.contains('^Unnamed')]
                dataTest =  dataTest.drop(['Pred', 'Pred2', 'h', 'e', 'u','ID'], axis = 1)
                dataTest = dataTest.loc[:, dataTest.var()!=0.0]

            except:
                print('---')

            names = dataTrain.columns
            scaler = preprocessing.StandardScaler()
            dataTrain = scaler.fit_transform(dataTrain)
            dataTrain = pd.DataFrame(dataTrain, columns=names)
            pca = PCA(n_components=Components)
            pca.fit(dataTrain)
            dataTrain = pca.transform(dataTrain)
            dataTrain = pd.DataFrame(dataTrain)
            dataTrain = ut.Polinomial(dataTrain,k)


            dataTest = scaler.fit_transform(dataTest)
            dataTest = pd.DataFrame(dataTest, columns=names)
            dataTest = pca.transform(dataTest)
            dataTest = pd.DataFrame(dataTest)
            dataTest = ut.Polinomial(dataTest,k)


            dataTrain = dataTrain.assign(label=yTrain.values)
            #dataTrain['label'] = yTrain
            List = list(dataTrain.columns)
            print(List)
            List.remove('label')
            myList = '+'.join(map(str, List))
            label = 'label~'
            function = ''.join((label, myList))
            print(function)
            priorsDict = {}
            start_post = timeit.default_timer()

            print(dataTrain.shape)
            print(yTrain.shape)
            print(dataTrain.head())
            print(yTrain.head())
            with pm.Model() as model:
                priorsDict['Intercept'] = pm.Flat.dist() #set 1: 0,1
                for j in range(len(List)):
                    print(List[j])
                    priorsDict[List[j]] =   pm.Flat.dist()
                pm.glm.GLM.from_formula(function, dataTrain, priors= priorsDict,
                                        family= pm.glm.families.Binomial())
            trace, model, map_ = bm.fitBayesianModel(model, yTrain = yTrain, method=7,
                                                 n_=80000, MAP = False,
                                                 jobs  = 2, star = 'rrlyr', classifier ='RL',
                                                 PCA = False)




            r = ut.get_z(dataTrain, trace = trace, model=model, burn_in = 500)
            predictions_1_Train = (ut.logistic_function_(r).mean(axis=1)>0.5).astype(int)
            print(predictions_1_Train)
            accTrain = accuracy_score(yTrain, predictions_1_Train, normalize=True)
            f1Train = f1_score(yTrain, predictions_1_Train, pos_label = 1)
            print('Accuracy train: ', accTrain)
            print('F1 score Train: ', f1Train)


            stop_post = timeit.default_timer()
            time_post = stop_post - start_post
            r = ut.get_z(dataTest, trace = trace, model=model, burn_in = 500)
            predictions_1_Test = (ut.logistic_function_(r).mean(axis=1)>0.5).astype(int)
            print(predictions_1_Test)
            accTest = accuracy_score(yTest, predictions_1_Test, normalize=True)
            f1Test = f1_score(yTest, predictions_1_Test, pos_label = 1)
            if plot:
                cm = confusion_matrix(yTest, predictions_1_Test)
                ut.plot_confusion_matrix(cm, ['all', 'rrlyr'], type = 'test')
            print('Accuracy test: ', accTest)
            print('F1 score Test: ', f1Test)

            time_ml_i = []
            i = 'rrlyr'
            modeltoFit = 'LR'
            dim = 2
            kernel = False
            gelRub = pm.diagnostics.gelman_rubin(trace)
            ml = True
            print('gelRub: ', gelRub)
            #try:
            start_2 = timeit.default_timer()
            logml_dict = bs.Marginal_llk(trace, model=model, maxiter=100000)
            print('Estimated Marginal log-Likelihood %.5f'%(logml_dict['logml']))
            marginal_likelihood = logml_dict['logml']
            stop_2 = timeit.default_timer()
            time_ml = stop_2 - start_2
            print('exporting model')



            time_cv = 0
            res.append([k, Components, marginal_likelihood, size, informative, gelRub, accTrain, accTest,
            f1Train, f1Test, time_cv, time_post])
            pd.DataFrame(res).to_csv('Results/summaryMCMC/'+'MarginalLikelihood_flat_3.csv')
            del dataTrain
            del dataTest
            print('___________________________________________________________________________')
