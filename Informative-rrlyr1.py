import pandas as pd
import timeit
import pymc3 as pm
import sys
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing
sys.path.insert(0, './src')
import bridgeSampling as bs
import utilFunctions as ut
import BayesianModels as bm

fileTrain = '/home/franciscoperez/Documents/GitHub/data/BIASEDFATS/Train_rrlyr-1.csv'
fileTest = '/home/franciscoperez/Documents/GitHub/data/BIASEDFATS/Test_rrlyr-1.csv'

dataTrain = pd.read_csv(fileTrain)
dataTest = pd.read_csv(fileTest)
DataPriors = [dataTrain, dataTest]
DataPriors = pd.concat(DataPriors)
lc_train = ut.downSampling(DataPriors)
RelevantFeatures = ['Amplitude', 'PeriodLS', 'label']
DataPriors = ut.MostImportanFeature(lc_train, RelevantFeatures)
DataPriors['newlabel'] = (DataPriors['Amplitude'] >= 0.2) & (DataPriors['Amplitude'] <= 0.8) & (
            DataPriors['PeriodLS'] >= 0.2) & (DataPriors['PeriodLS'] <= 1.0)
del DataPriors['label']
DataPriors.rename(columns={'newlabel': 'label'}, inplace=True)

res = []
PCA_ = True

dataTrain = pd.read_csv(fileTrain)
dataTest = pd.read_csv(fileTest)
DataPriors = dataTest

DataPriors['newlabel'] = (DataPriors['Amplitude'] >= 0.2) & (DataPriors['Amplitude'] <= 0.8) & (
            DataPriors['PeriodLS'] >= 0.2) & (DataPriors['PeriodLS'] <= 1.0)
del DataPriors['label']
DataPriors.rename(columns={'newlabel': 'label'}, inplace=True)
yPrior = DataPriors['label']
del DataPriors['label']

try:
    DataPriors = DataPriors.loc[:, ~DataPriors.columns.str.contains('^Unnamed')]
    DataPriors = DataPriors.drop(['Pred', 'Pred2', 'h', 'e', 'u', 'ID'], axis=1)
    DataPriors = DataPriors.loc[:, DataPriors.var() != 0.0]
except:
    print('---')

names = DataPriors.columns
scaler = preprocessing.StandardScaler()
DataPriors = scaler.fit_transform(DataPriors)
DataPriors = pd.DataFrame(DataPriors, columns=names)

Flat = False
for k in [2]:
    for Components in [10]:

        pca = PCA(n_components=Components)
        pca.fit(DataPriors)
        XPrior = pca.transform(DataPriors)
        XPrior = pd.DataFrame(XPrior)
        XPrior = ut.polynomial(XPrior, k)
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(XPrior, yPrior)
        intercept = clf.intercept_
        priors = clf.coef_[0]

        for size in [1000]:
            dataTrain = pd.read_csv(fileTrain)
            dataTest = pd.read_csv(fileTest)
            dataTrain = ut.downSampling(dataTrain)
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
        dataTrain = scaler.fit_transform(dataTrain)
        dataTrain = pd.DataFrame(dataTrain, columns=names)
        pca = PCA(n_components=Components)
        pca.fit(dataTrain)
        dataTrain = pca.transform(dataTrain)
        dataTrain = pd.DataFrame(dataTrain)
        dataTrain = ut.polynomial(dataTrain, k)

        names = dataTest.columns
        scaler = preprocessing.StandardScaler()
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

        priorsDict = {}

        start_post = timeit.default_timer()
        with pm.Model() as model:
            priorsDict['Intercept'] = pm.Normal.dist(mu=intercept[0], sd=1)
            for j in range(len(List)):
                priorsDict[List[j]] = pm.Normal.dist(mu=priors[j], sd=10)
            pm.glm.GLM.from_formula(function, dataTrain, priors=priorsDict,
                                    family=pm.glm.families.Binomial())

        trace, model, map_ = bm.fitBayesianModel(model, yTrain=yTrain, method=7,
                                                 n_=50000, MAP=False,
                                                 jobs=1, chains=2, star='rrlyr', classifier='RL',
                                                 PCA=False)
        trace = trace[500:]
        pm.traceplot(trace)
        print('plotting trace')

        stop_post = timeit.default_timer()
        time_post = stop_post - start_post

        r = ut.get_z(dataTrain, trace=trace, model=model, burn_in=500)
        predictions_1_Train = (ut.logistic_function_(r).mean(axis=1) > 0.5).astype(int)
        accTrain = accuracy_score(yTrain, predictions_1_Train, normalize=True)
        f1Train = f1_score(yTrain, predictions_1_Train, pos_label=1)
        print('Accuracy train: ', accTrain)
        print('Accuracy f1 Train: ', f1Train)

        r = ut.get_z(dataTest, trace=trace, model=model, burn_in=500)
        predictions_1_Test = (ut.logistic_function_(r).mean(axis=1) > 0.5).astype(int)

        accTest = accuracy_score(yTest, predictions_1_Test, normalize=True)
        f1Test = f1_score(yTest, predictions_1_Test, pos_label=1)
        print('Accuracy test: ', accTest)
        print('F1 score Test: ', f1Test)

        gelRub = pm.diagnostics.gelman_rubin(trace)
        print('gelRub: ', gelRub)
        try:
            start_2 = timeit.default_timer()
            logml_dict = bs.Marginal_llk(trace, model=model, maxiter=100000)
            print('Estimated Marginal log-Likelihood %.5f' % (logml_dict['logml']))
            marginal_likelihood = logml_dict['logml']
            stop_2 = timeit.default_timer()
            time_ml = stop_2 - start_2
        except:
            print('marginal likelihood does not estimated')
            marginal_likelihood = 'Null'
        print('exporting model')
        res.append([k, Components, marginal_likelihood, size, gelRub, accTrain, accTest,
                    f1Train, f1Test])
        pd.DataFrame(res).to_csv('Informative_rrlyr3-0707.csv')
        del dataTrain
        del dataTest
        print('___________________________________________________________________________')
