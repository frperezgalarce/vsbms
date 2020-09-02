import pandas as pd
import bridgeSampling as bs
import utilFunctions as ut
import BayesianModels as bm
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import timeit
import pymc3 as pm


def runExperimentsBiased(ml = True, size = [100], components = [8], method = [7], classes = ['rrlyr'],
                    fit_iterations = 20000, id_col_= 'ID', name_class_col_= 'Class', biasedSplit = False, ModifiedPrior=False, alpha_ = 0.1,
                    onetoOne = True, DropEasy_ = True, priors_ = 'normal', oneToOne_ = False, PCA_ = True, modeltoFit = 'RL', kernel = False,
                    poli = 3, n_hidden_ = 4, njobs= 16, **kwargs):

    print('PCA: '+str(PCA_))
    marginal_likelihood = 0
    res = []
    dim = components[0]
    print(dim)
    for i in classes:
        print('Class: ', i)
        dataTrain = pd.read_csv('data/BIASEDFATS/Train_OGLE_'+i+'.csv')
        dataTest = pd.read_csv('data/BIASEDFATS/Test_OGLE_'+i+'.csv')
        time_ml_i = []

        dataTrain = ut.down_sampling(dataTrain)
        samples = dataTrain.shape[0]

        maxSample = size[0]
        if samples > maxSample:
            samples = maxSample

            dataTrain = dataTrain.sample(samples, random_state =0)
            print('after down_sampling: ')
            '''print('train: ')
            print(data_train.label.value_counts())
            print('test: ')
            print(data_test.label.value_counts())
            '''
        print('The dataset contains:', samples, 'samples')

        try:
            del dataTrain['Unnamed: 0']
            del dataTest['Unnamed: 0']
            dataTrain = dataTrain.loc[:, ~dataTrain.columns.str.contains('^Unnamed')]
            dataTest = dataTest.loc[:, ~dataTest.columns.str.contains('^Unnamed')]
            del dataTrain['ID']
            del dataTest['ID']

            yTrain = dataTrain['label']
            yTest = dataTest['label']

            del dataTrain['label']
            del dataTest['label']

            try:
                dataTrain =  dataTrain.drop(['Pred', 'Pred2', 'h', 'e', 'u'], axis = 1)
                dataTest =  dataTest.drop(['Pred', 'Pred2', 'h', 'e', 'u'], axis = 1)
                dataTrain = dataTrain.loc[:, dataTrain.var()!=0.0]
                dataTest = dataTest.loc[:, dataTest.var()!=0.0]
            except:
                print('---')
        except:
            print('---')

        mostImportantF = True
        if PCA_ == False:
            print('PCA False')
            if mostImportantF == True:
                c_comp = 0
                RelevantFeatures = ['PeriodLS','CAR_tau','CAR_mean', 'CAR_sigma','Meanvariance', 'Skew', 'PercentDifferenceFluxPercentile','Gskew',
                'Class_col', 'Psi_CS', 'Psi_eta','SlottedA_length', 'RCs']
                xTrain =  ut.most_important_features(dataTrain, RelevantFeatures)
                xTest  =  ut.most_important_features(dataTest, RelevantFeatures)
            else:
                xTrain = dataTrain
                xTest = dataTest
            print('Running without dimentional reduction, forget argument components')
            print('before: ', xTrain.head())
            xTrain=(xTrain-xTrain.mean())/xTrain.std()
            xTest=(xTest-xTest.mean())/xTest.std()
            print('after: ')
            print(xTrain.head())

            if kernel == True:
                xTest = ut.kernelPolinomial(np.asanyarray(xTest),poli)
                xTrain = ut.kernelPolinomial(np.asanyarray(xTrain),poli)

        else:
            c_comp = components[0]
            print('compnents: ', c_comp)
            dataTrain=(dataTrain-dataTrain.mean())/dataTrain.std()
            dataTest=(dataTest-dataTest.mean())/dataTest.std()
            xTrain, yTrain = ut.dim_reduction(dataTrain, yTrain, c_comp)
            xTest, yTest = ut.dim_reduction(dataTest, yTest, c_comp)
            if kernel == True:
                xTest = ut.kernelPolinomial(xTest,poli)
                xTrain = ut.kernelPolinomial(xTrain,poli)
            else:
                xTest = pd.DataFrame(data = xTest, columns = ['PC'+str(i) for i in range(c_comp)])
                xTrain = pd.DataFrame(data = xTrain, columns = ['PC'+str(i) for i in range(c_comp)])

        xTrain['Class'] = yTrain.values
        xTest['Class']  = yTest.values
        DataTest, DataTrain = xTest, xTrain
        del xTest
        del xTrain

        acc_kfold_Train = []
        f1_kfold_Train = []

        skf = StratifiedKFold(n_splits=int(5))
        skf.get_n_splits(DataTrain, yTrain)
        start_1 = timeit.default_timer()


        for train_index, test_index in skf.split(DataTrain, yTrain):
            X_train, X_test = DataTrain.iloc[train_index,:], DataTrain.iloc[test_index,:]
            y_train, y_test = yTrain.iloc[train_index], yTrain.iloc[test_index]
            print('y_train')
            print((y_train.head()))


            model = bm.LogisticRegressionBinomialPrior(X_train, var_label1=kwargs['class_1'], var_label2=kwargs['class_2'],
                                               biasedSplit = biasedSplit, onetoOne = onetoOne, priors = priors_,
                                               className = name_class_col_, PCA =PCA_)



            trace, model, map = bm.fitbayesianmodel(model, ytrain= y_train, method=method[0],
                                                    n_=int(fit_iterations/njobs), MAP = False,
                                                    jobs  = njobs, star = i, classifier =modeltoFit,
                                                    PCA = PCA_)

            r = ut.get_z(X_train, trace = trace, model=model, burn_in = 500)
            predictions_1_Train = (ut.logistic_function_(r).mean(axis=1)>0.5).astype(int)

            y_train  = 1*(y_train == 'class_a')
            accTrain = accuracy_score(y_train, predictions_1_Train, normalize=True)
            f1Train = f1_score(y_train, predictions_1_Train, pos_label = 1)
            cm = confusion_matrix(y_train, predictions_1_Train)
            print('Accuracy train: ', accTrain)
            print('Accuracy f1 train: ', f1Train)
            acc_kfold_Train.append(accTrain)
            f1_kfold_Train.append(f1Train)

        accTrain = np.mean(acc_kfold_Train)
        f1Train =np.mean(f1Train)
        print('Mean Accuracy train: ', accTrain)
        print('Mean f1 train: ', f1Train)
        stop_1 = timeit.default_timer()
        time_CV = stop_1 - start_1



        start_post = timeit.default_timer()
        model = bm.LogisticRegressionBinomialPrior(DataTrain,
                                            var_label1=kwargs['class_1'], var_label2=kwargs['class_2'],
                                           biasedSplit = biasedSplit, onetoOne = onetoOne, priors = priors_,
                                           className = name_class_col_, PCA =PCA)
        trace, model, map = bm.fitbayesianmodel(model, ytrain= yTrain, method=method[0],
                                                n_=int(fit_iterations/njobs), MAP = False,
                                                jobs  = njobs, star = i, classifier =modeltoFit,
                                                PCA = PCA_)
        stop_post = timeit.default_timer()
        time_post = stop_post - start_post

        del DataTest['Class']
        r = ut.get_z(DataTest, trace = trace, model=model, burn_in = 500)
        predictions_1_Test = (ut.logistic_function_(r).mean(axis=1)>0.5).astype(int)
        yTest  = 1*(yTest == 'class_a')
        accTest = accuracy_score(yTest, predictions_1_Test, normalize=True)
        f1Test = f1_score(yTest, predictions_1_Test, pos_label = 1)
        print('Accuracy train: ', accTest)
        print('Accuracy f1 train: ', f1Test)

        gelRub = pm.diagnostics.gelman_rubin(trace)
        print('gelRub: ', gelRub)
        try:
             if(ml == True):
                 start_2 = timeit.default_timer()
                 logml_dict = bs.Marginal_llk(trace, model=model, maxiter=100000)
                 print('Estimated Marginal log-Likelihood %.5f'%(logml_dict['logml']))
                 marginal_likelihood = logml_dict['logml']
                 stop_2 = timeit.default_timer()
                 time_ml = stop_2 - start_2
                 try:
                     print('WAIC Estimation')
                     RLB_waic  = pm.waic(trace, model)
                     waic = RLB_waic.WAIC
                     print(waic)
                 except:
                     waic = 0

                 listData = [c_comp, poli, method, time_post, time_ml, marginal_likelihood, waic, gelRub, accTrain, accTest,
                 f1Train, f1Test]
                 time_ml_i.append(listData)
             else:
                 time_ml_i.append([c_comp, 'null', time_post, 'null', 'null', 'null', 'null', gelRub, accTrain, accTest,
                 f1Train, f1Test])
        except:
            print('marginal likelihood does not estimated')
        print('exporting model')
        df = pd.DataFrame(np.asanyarray(time_ml_i))
        print(df.head())
        df.to_csv('Results/summaryMCMC/0309/'+'dataAnalysis_Features_'+i+'_'+modeltoFit+'_'+str(c_comp)+'_'+str(dim)+'_'+str(kernel)+'_'+'.csv')
        print("return the last model and trace")
        res.append([marginal_likelihood, model, trace, map, i, gelRub, accTrain, accTest,
        f1Train, f1Test, time_CV, time_ml, time_post])
    return res
