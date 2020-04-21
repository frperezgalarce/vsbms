import pandas as pd
import bridgeSampling as bs # it contains a method to estimate the marginal likelihood according to the Bridge Sampling approach
import utilFunctions as ut          # it has different methods to handle and plot data
import BayesianModels as bm # it has methods to build and train bayesian model (Logistic Regression and Neural Nets)

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
#import seaborn
import matplotlib.pyplot as plt
from itertools import cycle
from warnings import filterwarnings
from sklearn.model_selection import StratifiedKFold

import timeit
import pymc3 as pm
import sys, os

# def runExperiments(dataSet, ml = True, size = [0.2], components = [8], method = [4],
#                     fit_iterations = 20000, id_col_= 'ID', name_class_col_= 'Class', biasedSplit = False, ModifiedPrior=False, alpha_ = 0.1,
#                     onetoOne = True, DropEasy_ = True, priors_ = 'student', oneToOne_ = True, PCA_ = True, modeltoFit = 'RL', kernel = True,
#                     poli = 3, n_hidden_ = 2, njobs= 16, **kwargs):
#     ' dataSet: Pandas object, ml : boolen (if is estimated the marginal likelihood)'
#
#     time_ml_i = []
#     for s in size:
#         xTrain, xTest, yTrain, yTest = ut.Define_TrainSet(dataSet, plot= False, test=(1-s), biasedSplit = biasedSplit,
#                                                                     alpha = alpha_, DropEasy = DropEasy_,oneToOne = oneToOne_, **kwargs)
#         print('Running experiments')
#
#         print('Train shape: ', xTrain.shape)
#         print('Test shape: ', xTest.shape)
#
#         if biasedSplit == True:
#             del xTrain[id_col_]
#             del xTest[id_col_]
#
#
#         train = ut.preprocess(xTrain)
#         xTrain = train.replace('\n', '', regex = True).replace('null', '0.0', regex = True).apply(pd.to_numeric, errors ='ignore')
#
#         test = ut.preprocess(xTest)
#         xTest = test.replace('\n', '', regex = True).replace('null', '0.0', regex = True).apply(pd.to_numeric, errors ='ignore')
#
#         for c_comp in components:
#              #print(xTrain.columns)
#             if PCA_ == True:
#                  print(c_comp)
#                  np_scaled = ut.preprocessing.normalize(xTrain)
#                  df_normalized = pd.DataFrame(np_scaled)
#                  pca = PCA(n_components=c_comp)
#                  principalComponents = pca.fit_transform(df_normalized)
#                  if kernel == True:
#                      principalComponents = ut.kernelPolinomial(principalComponents,poli)
#                      Reduced_Df = pd.DataFrame(data = principalComponents, columns = ['PC'+str(i) for i in range(c_comp*poli)])
#                  else:
#                      Reduced_Df = pd.DataFrame(data = principalComponents, columns = ['PC'+str(i) for i in range(c_comp)])
#                  yTrain.reset_index(drop=True, inplace=True)
#                  Reduced_Df.reset_index(drop=True, inplace=True)
#                  DataTrain = pd.concat([yTrain, Reduced_Df], axis=1, ignore_index=True)
#                  if kernel == True:
#                      col = ['PC'+str(i) for i in range(c_comp*poli)]
#                  else:
#                      col = ['PC'+str(i) for i in range(c_comp)]
#                  col.insert(0, name_class_col_)
#                  DataTrain.columns = col
#
#                  np_scaled_test = ut.preprocessing.normalize(xTest)
#                  df_normalized_test = pd.DataFrame(np_scaled_test)
#                  principalComponents_test = pca.fit_transform(df_normalized_test)
#
#                  if kernel == True:
#                      principalComponents_test = ut.kernelPolinomial(principalComponents_test,poli)
#                      Reduced_Df_test = pd.DataFrame(data = principalComponents_test, columns = ['PC'+str(i) for i in range(c_comp*poli)])
#                  else:
#                      Reduced_Df_test = pd.DataFrame(data = principalComponents_test, columns = ['PC'+str(i) for i in range(c_comp)])
#
#                  yTest.reset_index(drop=True, inplace=True)
#                  Reduced_Df_test.reset_index(drop=True, inplace=True)
#                  DataTest = pd.concat([yTest, Reduced_Df_test], axis=1, ignore_index=True)
#
#                  if kernel == True:
#                      col = ['PC'+str(i) for i in range(c_comp*poli)]
#                  else:
#                      col = ['PC'+str(i) for i in range(c_comp)]
#
#                  col.insert(0, name_class_col_)
#                  DataTest.columns = col
#                  print(DataTrain.columns)
#             else:
#                 print('Running without dimentional reduction, forget argument components')
#                 col = xTrain.columns
#                 print(col.shape)
#                 print(xTrain.shape)
#                 np_scaled = ut.preprocessing.normalize(xTrain)
#                 print(np_scaled.shape)
#                 train = pd.DataFrame(np_scaled, columns = col)
#                 yTrain.reset_index(drop=True, inplace=True)
#                 train.reset_index(drop=True, inplace=True)
#                 DataTrain =  pd.concat([yTrain, train], axis=1, ignore_index=True)
#                 col = list(train.columns)
#                 #print(col.shape)
#                 print(col)
#                 col.insert(0, name_class_col_)
#                 #print(col.shape)
#                 DataTrain.columns = col
#                 print(DataTrain.columns)
#
#                 col = xTest.columns
#                 np_scaled = ut.preprocessing.normalize(xTest)
#                 test = pd.DataFrame(np_scaled, columns = col)
#
#                 yTest.reset_index(drop=True, inplace=True)
#                 test.reset_index(drop=True, inplace=True)
#
#                 DataTest = pd.concat([yTest, test], axis=1, ignore_index=True)
#                 col = list(test.columns)
#                 col.insert(0, name_class_col_)
#                 DataTest.columns = col
#                 print(DataTrain.columns)
#                 print(name_class_col_)
#
#
#             if ModifiedPrior == True:
#                    if(modeltoFit == 'RL'):
#                          model, DataTrain, DataTest  = bm.LogisticRegressionBinomialPrior(DataTrain, DataTest, var_label1=kwargs['class_1'], var_label2=kwargs['class_2'], biasedSplit = biasedSplit, onetoOne = onetoOne, priors = priors_, className = name_class_col_, PCA =PCA)
#                    if(modeltoFit == 'NN'):
#                           print('running NN')
#                           model, DataTrain, DataTest  = bm.construct_nn(DataTrain, name_class_col_, n_hidden = n_hidden_, typeoutput = 'bernoulli', layers = 1, **kwarg)
#             else:
#                     if(modeltoFit == 'RL'):
#                          model, DataTrain, DataTest  = bm.LogisticRegressionBinomial(DataTrain, DataTest, biasedSplit = biasedSplit, onetoOne = onetoOne, className = name_class_col_, PCA =PCA, **kwargs)
#                     if(modeltoFit == 'NN'):
#                           print('running NN')
#                           model, DataTrain, DataTest  = bm.construct_nn(DataTrain, name_class_col_, n_hidden = n_hidden_, typeoutput = 'bernoulli', layers = 2, **kwarg)
#             start_1 = timeit.default_timer()
#
#             for meth in method:
#                  trace, model = bm.fitBayesianModel(model, method=meth, n_=fit_iterations, MAP = True, jobs  = njobs)
#                  stop_1 = timeit.default_timer()
#                  time_post = stop_1 - start_1
#                  print(modeltoFit)
#
#                  if(modeltoFit == 'RL'):
#                      r = ut.get_z(DataTrain, trace = trace, model = model, burn_in = 500)
#                      predictions_1_Train = ut.logistic_function_(r).mean(axis=1)>0.5
#                      accTrain = accuracy_score(yTrain, predictions_1_Train, normalize=True)
#                      f1Train = f1_score(yTrain, predictions_1_Train, normalize=True, pos_label = 'ClassA')
#                      cm = confusion_matrix(yTrain, predictions_1_Train)
#                      ut.plot_confusion_matrix(cm, ['ClassA', 'ClassB'])
#                      print('Accuracy train: ', accTrain)
#                      print('Accuracy f1 Train: ', f1Train)
#
#                      r = ut.get_z(DataTest, trace = trace, model=model, burn_in = 500)
#                      predictions_1_Test = ut.logistic_function_(r).mean(axis=1)>0.5
#                      accTest = accuracy_score(yTest, predictions_1_Test, normalize=True)
#                      f1Test = f1_score(yTest, predictions_1_Test, normalize=True, pos_label = 'ClassA')
#                      print('Accuracy train: ', accTest)
#                      print('Accuracy f1 Train: ', f1Test)
#
#                  else:
#                      accTrain = 0
#                      accTest = 0
#                      f1Test = 0
#                      f1Train = 0
#
#                  try:
#                      if(ml == True):
#                          start_2 = timeit.default_timer()
#                          logml_dict = bs.Marginal_llk(trace, model=model, maxiter=1000)
#                          print('Estimated Marginal log-Likelihood %.5f'%(logml_dict['logml']))
#                          stop_2 = timeit.default_timer()
#                          time_ml = stop_2 - start_2
#                          if modeltoFit == 'RL':
#                              try:
#                                  print('WAIC Estimation')
#                                  RLB_waic  = pm.waic(trace, model)
#                                  waic = RLB_waic.WAIC
#                              except:
#                                  waic = 0
#                          else:
#                             waic = 'NULL'
#                          print('saving statistics')
#                          time_ml_i.append([c_comp, meth, time_post, time_ml, logml_dict['logml'], waic, s, accTrain, accTest,
#                          f1Test, f1Train])
#                      else:
#                          time_ml_i.append([c_comp, time_post, trace, accTrain, accTest, f1Test, f1Train])
#                  except:
#                      print('marginal likelihood does not estimated')
#     print('exporting model')
#     pd.DataFrame(time_ml_i).to_csv('dataAnalysis_time_size.csv')
#     print("return the last model and trace")
#     return pd.DataFrame(time_ml_i), model, trace, DataTrain, DataTest, accTrain, accTest, f1Test, f1Train

def runExperimentsBiased(ml = True, size = [100], components = [8], method = [7], classes = ['rrlyr'],
                    fit_iterations = 20000, id_col_= 'ID', name_class_col_= 'Class', biasedSplit = False, ModifiedPrior=False, alpha_ = 0.1,
                    onetoOne = True, DropEasy_ = True, priors_ = 'normal', oneToOne_ = False, PCA_ = True, modeltoFit = 'RL', kernel = False,
                    poli = 3, n_hidden_ = 4, njobs= 16, **kwargs):

    ' dataSet: Pandas object, ml : boolen (if is estimated the marginal likelihood)'
    #classes = ['dpv', 'dsct',  'lpv', 'acv',  'rcb','rrlyr','t2cep', 'wd', 'yso','acep','cep','dn']
    #, 'dsct','ecl',  'lpv', 'cep'
    print('PCA: '+str(PCA_))
    marginal_likelihood = 0
    res = []
    dim = components[0]
    print(dim)
    for i in classes:
        print('Class: ', i)
        dataTrain = pd.read_csv('data/BIASEDFATS/Train_OGLE_'+i+'.csv')
        dataTest = pd.read_csv('data/BIASEDFATS/Test_OGLE_'+i+'.csv')
        #dataTrain = pd.read_csv('data/vae_features/Train_OGLE_'+i+'_lat_'+str(dim)+'_vae.csv')
        #dataTest = pd.read_csv('data/vae_features/Test_OGLE_'+i+'_lat_'+str(dim)+'_vae.csv')

        time_ml_i = []

        dataTrain = ut.downSampling(dataTrain)
        samples = dataTrain.shape[0]
        #print(dataTrain.head(30))

        maxSample = size[0]
        if samples > maxSample:
            samples = maxSample

            dataTrain = dataTrain.sample(samples, random_state =0)
            print('after downSampling: ')
            '''print('Train: ')
            print(dataTrain.label.value_counts())
            print('Test: ')
            print(dataTest.label.value_counts())
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
                xTrain =  ut.MostImportanFeature(dataTrain, RelevantFeatures)
                xTest  =  ut.MostImportanFeature(dataTest, RelevantFeatures)
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
            xTrain, yTrain = ut.DimReduction(dataTrain, yTrain, c_comp)
            xTest, yTest = ut.DimReduction(dataTest, yTest, c_comp)
            if kernel == True:
                xTest = ut.kernelPolinomial(xTest,poli)
                xTrain = ut.kernelPolinomial(xTrain,poli)
            else:
                xTest = pd.DataFrame(data = xTest, columns = ['PC'+str(i) for i in range(c_comp)])
                xTrain = pd.DataFrame(data = xTrain, columns = ['PC'+str(i) for i in range(c_comp)])


        #print('1', xTrain)
        #print(type(yTrain))
        xTrain['Class'] = yTrain.values
        #print('2', xTrain)
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
            #print(train_index)
            #print(test_index)
            X_train, X_test = DataTrain.iloc[train_index,:], DataTrain.iloc[test_index,:]
            y_train, y_test = yTrain.iloc[train_index], yTrain.iloc[test_index]

            # print((X_train.head()))
            print('y_train')
            print((y_train.head()))


            model = bm.LogisticRegressionBinomialPrior(X_train, var_label1=kwargs['class_1'], var_label2=kwargs['class_2'],
                                               biasedSplit = biasedSplit, onetoOne = onetoOne, priors = priors_,
                                               className = name_class_col_, PCA =PCA_)



            trace, model, map = bm.fitBayesianModel(model, yTrain = y_train, method=method[0],
                                         n_=int(fit_iterations/njobs), MAP = False,
                                         jobs  = njobs, star = i, classifier =modeltoFit,
                                         PCA = PCA_)

            r = ut.get_z(X_train, trace = trace, model=model, burn_in = 500)
            predictions_1_Train = (ut.logistic_function_(r).mean(axis=1)>0.5).astype(int)

            y_train  = 1*(y_train == 'ClassA')
            accTrain = accuracy_score(y_train, predictions_1_Train, normalize=True)
            f1Train = f1_score(y_train, predictions_1_Train, pos_label = 1)
            cm = confusion_matrix(y_train, predictions_1_Train)
            #ut.plot_confusion_matrix(cm, [i, 'all'], type = 'train')
            print('Accuracy train: ', accTrain)
            print('Accuracy f1 Train: ', f1Train)
            acc_kfold_Train.append(accTrain)
            f1_kfold_Train.append(f1Train)

        accTrain = np.mean(acc_kfold_Train)
        f1Train =np.mean(f1Train)
        print('Mean Accuracy train: ', accTrain)
        print('Mean f1 Train: ', f1Train)
        stop_1 = timeit.default_timer()
        time_CV = stop_1 - start_1



        start_post = timeit.default_timer()
        model = bm.LogisticRegressionBinomialPrior(DataTrain,
                                            var_label1=kwargs['class_1'], var_label2=kwargs['class_2'],
                                           biasedSplit = biasedSplit, onetoOne = onetoOne, priors = priors_,
                                           className = name_class_col_, PCA =PCA)
        trace, model, map = bm.fitBayesianModel(model, yTrain = yTrain, method=method[0],
                                     n_=int(fit_iterations/njobs), MAP = False,
                                     jobs  = njobs, star = i, classifier =modeltoFit,
                                     PCA = PCA_)
        stop_post = timeit.default_timer()
        time_post = stop_post - start_post

        del DataTest['Class']
        r = ut.get_z(DataTest, trace = trace, model=model, burn_in = 500)
        predictions_1_Test = (ut.logistic_function_(r).mean(axis=1)>0.5).astype(int)
        yTest  = 1*(yTest == 'ClassA')
        accTest = accuracy_score(yTest, predictions_1_Test, normalize=True)
        f1Test = f1_score(yTest, predictions_1_Test, pos_label = 1)
        #cm = confusion_matrix(yTest, predictions_1_Test)
        #ut.plot_confusion_matrix(cm, [i, 'all'], type = 'test')
        print('Accuracy train: ', accTest)
        print('Accuracy f1 Train: ', f1Test)

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

                 print(time_ml_i)
                 print('saving statistics')
                 print(c_comp)
                 print(method)
                 print(time_post)
                 print(time_ml)
                 print(marginal_likelihood)
                 print(waic)
                 print(gelRub)
                 print(accTrain)
                 print(accTest)
                 print(f1Train)
                 print(f1Test)

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
        #res.to_csv('Results/summaryMCMC/0309/'+'dataAnalysis3_'+i+'_'+modeltoFit+'_'+str(c_comp)+'_'+str(dim)+'_'+str(kernel)+'_'+'.csv')
    return res
