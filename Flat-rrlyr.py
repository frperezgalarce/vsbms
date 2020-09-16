import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import timeit
import pymc3 as pm
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

sys.path.insert(0, './src')
import bridgeSampling as bs
import utilFunctions as ut
import BayesianModels as bm

fileTrain = '/home/franciscoperez/Documents/GitHub/vsbms/data/BIASEDFATS/Train_rrlyr-1.csv'
fileTest = '/home/franciscoperez/Documents/GitHub/vsbms/data/BIASEDFATS/Test_rrlyr-1.csv'

res = []
for k in [2, 3]:
    for Components in [8, 10, 12]:
        for size in [500]:
            dataTrain = pd.read_csv(fileTrain)
            dataTest = pd.read_csv(fileTest)
            dataTrain = ut.down_sampling(dataTrain)
            try:
                dataTrain = dataTrain.sample(size, random_state=0)
            except:
                print('sample bigger than data size')

            yTrain = 1 * (dataTrain['label'] == 'ClassA')

            del dataTrain['label']
            yTest = 1 * (dataTest['label'] == 'ClassA')
            del dataTest['label']

            try:
                dataTrain = dataTrain.loc[:, ~dataTrain.columns.str.contains('^Unnamed')]
                dataTrain = dataTrain.drop(['Pred', 'Pred2', 'h', 'e', 'u', 'ID'], axis=1)
                dataTrain = dataTrain.loc[:, dataTrain.var() != 0.0]

                dataTest = dataTest.loc[:, ~dataTest.columns.str.contains('^Unnamed')]
                dataTest = dataTest.drop(['Pred', 'Pred2', 'h', 'e', 'u', 'ID'], axis=1)
                dataTest = dataTest.loc[:, dataTest.var() != 0.0]
            except:
                print('---')

            names = dataTrain.columns
            scaler = preprocessing.StandardScaler()
	        print(dataTrain.shape)
            dataTrain = scaler.fit_transform(dataTrain)
            dataTrain = pd.DataFrame(dataTrain, columns=names)
            pca = PCA(n_components=Components)
            pca.fit(dataTrain)
            dataTrain = pca.transform(dataTrain)
            dataTrain = pd.DataFrame(dataTrain)
            dataTrain = ut.polynomial(dataTrain, k)

            dataTest = scaler.fit_transform(dataTest)
            dataTest = pd.DataFrame(dataTest, columns=names)
            dataTest = pca.transform(dataTest)
            dataTest = pd.DataFrame(dataTest)
            dataTest = ut.polynomial(dataTest, k)

            dataTrain = dataTrain.assign(label=yTrain.values)
            List = list(dataTrain.columns)
            List.remove('label')
            myList = '+'.join(map(str, List))
            label = 'label~'
            function = ''.join((label, myList))
            print(function)
            priorsDict = {}
            start_post = timeit.default_timer()
            with pm.Model() as model:
                priorsDict['Intercept'] = pm.Flat.dist()  # set 1: 0,1
                for j in range(len(List)):
                    priorsDict[List[j]] = pm.Flat.dist()
                pm.glm.GLM.from_formula(function, dataTrain, priors=priorsDict,
                                        family=pm.glm.families.Binomial())
            trace, model, map_ = bm.fitbayesianmodel(model, ytrain=yTrain, method=7,
                                                     n_=20000, MAP=False,
                                                     jobs=1, star='rrlyr', classifier='RL',
                                                     PCA=False)

            trace = trace[500:]
            r = ut.get_z(dataTrain, trace=trace, burn_in=500)
            predictions_1_Train = (ut.logistic_function_(r).mean(axis=1) > 0.5).astype(int)
            accTrain = accuracy_score(yTrain, predictions_1_Train, normalize=True)
            f1Train = f1_score(yTrain, predictions_1_Train, pos_label=1)
            print('Accuracy train: ', accTrain)
            print('F1 score train: ', f1Train)

            stop_post = timeit.default_timer()
            time_post = stop_post - start_post
            r = ut.get_z(dataTest, trace=trace, burn_in=500)
            predictions_1_Test = (ut.logistic_function_(r).mean(axis=1) > 0.5).astype(int)

            accTest = accuracy_score(yTest, predictions_1_Test, normalize=True)
            f1Test = f1_score(yTest, predictions_1_Test, pos_label=1)
            print('Accuracy test: ', accTest)
            print('F1 score test: ', f1Test)

            time_ml_i = []
            gelRub = pm.diagnostics.gelman_rubin(trace)
            start_2 = timeit.default_timer()
            logml_dict = bs.Marginal_llk(trace, model=model, maxiter=200000)
            print('Estimated Marginal log-Likelihood %.5f' % (logml_dict['logml']))
            marginal_likelihood = logml_dict['logml']
            bm.error_measures(logml_dict)
            stop_2 = timeit.default_timer()
            time_ml = stop_2 - start_2
            res.append([k, Components, marginal_likelihood, size, gelRub, accTrain, accTest,
                        f1Train, f1Test, time_post])
            pd.DataFrame(res).to_csv('MarginalLikelihood_flat_1-500.csv')
            del dataTrain
            del dataTest
            print('___________________________________________________________________________')
