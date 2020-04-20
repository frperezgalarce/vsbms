
# coding: utf-8

# In[ ]:

#get_ipython().magic('matplotlib inline')
import theano
floatX = theano.config.floatX
theano.config.floatX = 'float64'
theano.config.compute_test_value ='raise'
theano.config.exception_verbosity = 'high'

import pymc3 as pm
import theano.tensor as T
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
#sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from pymc3.theanof import set_tt_rng, MRG_RandomStreams
from sklearn.metrics import confusion_matrix
import itertools
set_tt_rng(MRG_RandomStreams(42))
from pymc3.model import modelcontext
from scipy import dot
from scipy.linalg import cholesky as chol
import scipy.stats as st
from scipy import optimize
import warnings
from tempfile import mkdtemp
from pymc3.step_methods import smc
#import seaborn
from sklearn import preprocessing
from pymc3.variational.callbacks import CheckParametersConvergence
import timeit
from pymc3.variational.callbacks import CheckParametersConvergence
import sys
sys.setrecursionlimit(5000)
# In[ ]:

# def LogisticRegressionBinomial(DataTrain, DataTest, var_label1=0, var_label2 = 1, classRef = 'RRLYR', biasedSplit = False, onetoOne = True, className = 'Class', PCA = True, **kwargs):
#
#     if onetoOne == True: # and biasedSplit == False:
#         DataTrain = DataTrain[(DataTrain[className] == var_label1) | (DataTrain[className] == var_label2)]
#         DataTrain['label'] = 1*(DataTrain[className] == var_label1)
#         del DataTrain[className]
#
#         DataTest = DataTest[(DataTest[className] == var_label1) | (DataTest[className] == var_label2)]
#         DataTest['label'] = 1*(DataTest[className] == var_label1)
#         del DataTest[className]
#     else:
#         print('binary classification of', var_label1)
#         #Data = OGLE
#         DataTrain['label'] = 1*(DataTrain[className] == classRef)
#         del DataTrain[className]
#
#         DataTest['label'] = 1*(DataTest[className] == classRef)
#         del DataTest[className]
#
#     if PCA == True:
#         List = ['PC'+str(i) for i in range((DataTrain.shape[1])-1)]
#     else:
#         List = list(DataTrain.columns)
#         List.remove('label')
#
#     priorsDict = {}
#     priorsDict['Intercept'] = pm.Normal.dist(mu=0, sd=1)
#     for j in List:
#         priorsDict[j] =  pm.Normal.dist(mu=0, sd=1)
#
#
#     myList = '+'.join(map(str, List))
#     label = 'label~'
#     print(myList)
#     function = ''.join((label, myList))
#
#     with pm.Model() as logistic_model:
#         pm.glm.GLM.from_formula(function, DataTrain, priors= priorsDict, family=pm.glm.families.Binomial())
#         return logistic_model, DataTrain, DataTest


def LogisticRegressionBinomialPrior(DataTrain, var_label1=0, var_label2 = 1, biasedSplit = False, priors = 'normal', onetoOne = True, className = 'Class', PCA = True,**kwargs):
    '''
    if onetoOne == True:
        DataTrain = DataTrain[(DataTrain[className] == var_label1) | (DataTrain[className] == var_label2)]
        DataTrain['label'] = 1*(DataTrain[className] == var_label1)
        del DataTrain[className]

        DataTest = DataTest[(DataTest[className] == var_label1) | (DataTest[className] == var_label2)]
        DataTest['label'] = 1*(DataTest[className] == var_label1)
        del DataTest[className]
    '''
    #print(DataTrain)

    DataTrain['label']  = 1*(DataTrain[className] == 'ClassA')
    Y_train = DataTrain['label']
    del DataTrain[className]
    #del DataTest[className]


    print(Y_train)
    if PCA == True:
        List = ['PC'+str(i) for i in range(1, DataTrain.shape[1])]
    else:
        List = list(DataTrain.columns)
        try:
            List.remove('label')
        except:
            print('__')


    #

    print(List)
    myList = '+'.join(map(str, List))
    label = 'label~'
    function = ''.join((label, myList))
    print(function)

    priorsDict = {}
    vague = False
    if vague == True:
        print('Vague prior')
        with pm.Model() as logistic_model:
            mean1 = pm.Normal('mean1', mu=0, sd=1000) #1000,100, 10, 1
            if(priors == 'normal'):
                priorsDict['Intercept'] = pm.Normal.dist(mu=mean1, sd=1) #set 1: 0,1
                for j in List:
                    priorsDict[j] =  pm.Normal.dist(mu=mean1, sd=1)
            if(priors == 'student'):
                priorsDict['Intercept'] = pm.StudentT.dist(nu=1, mu=0, lam=1) # set 1: 15, 0, 1
                for j in List:
                    priorsDict[j] =  pm.StudentT.dist(nu=1, mu=0, lam=1)
            if(priorsDict == 'laplace'):
                priorsDict['Intercept'] = pm.laplace.dist(mu=0, b=1) # set 1: 0, 1
                for j in List:
                    priorsDict[j] =  pm.laplace.dist(mu=0, b=1)
            print(DataTrain.head())
            pm.glm.GLM.from_formula(function, DataTrain, priors= priorsDict, family= pm.glm.families.Binomial())
            return logistic_model

    else:
        print('Flat prior')
        with pm.Model() as logistic_model:
            if(priors == 'flat'):
                priorsDict['Intercept'] = pm.Flat.dist() # set 1: 0, 1
                for j in List:
                    priorsDict[j] =  pm.Flat.dist()

                print(DataTrain.head())
                pm.glm.GLM.from_formula(function, DataTrain, priors= priorsDict, family= pm.glm.families.Binomial())
                return logistic_model

def LogisticRegressionCategorical(Data):
    List = ['PC'+str(i) for i in range((Data.shape[1])-1)]
    myList = '+'.join(map(str, List))
    label = 'class_name~'
    function = ''.join((label, myList))
    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula(function, Data)
        return logistic_model

def construct_nn(Data_train, className, n_hidden = 8, activation = 'tanh', typeoutput = 'bernoulli', layers = 2, onetoOne=True, priors = 'normal',**kwargs):
    'xm = complete dataset only is used for defining the structure... replace it'

    print('shape: ', Data_train.shape)

    #Data_train = #Data_train[(Data_train[className] == kwargs['class_1']) | (Data_train[className] == kwargs['class_2'])]
    #Data_train['label'] = 1*(Data_train[className] == kwargs['class_1'])
    Data_train[className] = Data_train[className].map({'class_1': 1, 'class_2': 0})
    Y_train = Data_train[className]
    del Data_train[className]

    print('shape: ', Data_train.shape)

    try:
        del Data_train['label']
    except:
        print(Data_train.head())


    X_train = Data_train.astype(floatX)
    Y_train = Y_train.astype(floatX)

    ann_input = theano.shared(X_train.as_matrix())
    ann_output = theano.shared(Y_train.as_matrix())

    # Initialize random weights between each layer
    init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)

    print('shape: ', X_train.shape[1], n_hidden)

    if(layers == 2):
        init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)

    init_out = np.random.randn(n_hidden).astype(floatX)

    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        if(layers == 2):

            if(priors == 'normal'):
                weights_in_1 = pm.Normal('w_in_1', 0, sd=1, shape=(X_train.shape[1], n_hidden), testval=init_1)
                print(weights_in_1)
                # Weights from 1st to 2nd layer
                weights_1_2 = pm.Normal('w_1_2', 0, sd=1, shape=(n_hidden, n_hidden), testval=init_2)
                # Weights from hidden layer to output
                weights_2_out = pm.Normal('w_2_out', 0, sd=1, shape=(n_hidden,), testval=init_out)

            if(priors == 'student'):
                weights_in_1 = pm.StudentT('w_in_1', nu=15, mu=0, sd=1, shape=(X_train.shape[1], n_hidden), testval=init_1)
                # Weights from 1st to 2nd layer
                weights_1_2 = pm.StudentT('w_1_2', nu=15, mu=0, sd=1, shape=(n_hidden, n_hidden), testval=init_2)
                # Weights from hidden layer to output
                weights_2_out = pm.StudentT('w_2_out', nu=15, mu=0, sd=1, shape=(n_hidden,), testval=init_out)

            if(priors == 'laplace'):
                weights_in_1 = pm.laplace('w_in_1', mu=0, b=1, shape=(X_train.shape[1], n_hidden), testval=init_1)
                # Weights from 1st to 2nd layer
                weights_1_2 = pm.laplace('w_1_2', mu=0, b=1, shape=(n_hidden, n_hidden), testval=init_2)
                # Weights from hidden layer to output
                weights_2_out = pm.laplace('w_2_out', mu=0, b=1, shape=(n_hidden,), testval=init_out)


            # Build neural-network using tanh activation function
            if activation == 'relu':
                act_1 = theano.tensor.nnet.nnet.relu(pm.math.dot(ann_input, weights_in_1))
                act_2 = theano.tensor.nnet.nnet.relu(pm.math.dot(act_1, weights_1_2))
                act_out = theano.tensor.nnet.nnet.relu(pm.math.dot(act_2, weights_2_out))

            else:
                act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
                act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
                act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

        if(layers == 1):

            if(priors == 'normal'):
                weights_in_1 = pm.Normal('w_in_1', 0, sd=1, shape=(X_train.shape[1], n_hidden), testval=init_1)
                weights_2_out = pm.Normal('w_1_out', 0, sd=1, shape=(n_hidden,), testval=init_out)

            if(priors == 'student'):
                weights_in_1 = pm.StudentT('w_in_1', nu=15, mu=0, sd=1, shape=(X_train.shape[1], n_hidden), testval=init_1)
                weights_2_out = pm.StudentT('w_1_out', nu=15, mu=0, sd=1, shape=(n_hidden,), testval=init_out)

            if(priors == 'laplace'):
                weights_in_1 = pm.laplace('w_in_1', mu=0, b=1, shape=(X_train.shape[1], n_hidden), testval=init_1)
                weights_2_out = pm.laplace('w_1_out', mu=0, b=1, shape=(n_hidden,), testval=init_out)


            if activation == 'relu':
                act_1 = theano.tensor.nnet.nnet.relu(pm.math.dot(ann_input, weights_in_1))
                act_out = theano.tensor.nnet.nnet.relu(pm.math.dot(act_1, weights_2_out))
            else:
                act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
                act_out = pm.math.sigmoid(pm.math.dot(act_1, weights_2_out))

        # Binary classification -> Bernoulli likelihood

        if(typeoutput =='bernoulli'):
            out = pm.Bernoulli('out',
                               act_out,
                               observed=ann_output,
                               total_size=Y_train.shape[0]
                              )

        if(typeoutput == 'categorical'):
            out = pm.Categorical('out',
                           act_out,
                           observed=ann_output,
                           total_size=Y_train.shape[0]
                          )

    return neural_network, ann_input, ann_output


def fitBayesianModel(Bayesian_Model, yTrain, method = 1, n_ = 3000, MAP=True, chains =1, jobs =1, star = 'rrlyr', classifier ='RL', PCA = False):
    'This method fits bayesian model using different algorithms, option two only is able to work with NN'
#    get_ipython().magic('%time')

    print('chains: ', chains)
    print('jobs: ', jobs)
    if(method == 1):
        print('-------SGFS--------')
        with Bayesian_Model as model:
            if(MAP==True):
                map = pm.find_MAP()
                inference = pm.SGFS(batch_size = 50, total_size = len(yTrain))
                trace = pm.sample(start = map, draws = n_, step = inference)
                #inference = pm.ADVI()
                #approx = pm.fit(method=inference,start=start, callbacks=[CheckParametersConvergence(diff='absolute')])
                #approx = pm.fit(method='advi',start =map, callbacks=[CheckParametersConvergence(diff='absolute')])
            else:
                map = pm.find_MAP()
                inference = pm.SGFS(batch_size = 50, total_size = len(yTrain))
                #approx = pm.fit(method=inference, callbacks=[CheckParametersConvergence()])

                trace = pm.sample(draws = n_, step = inference)
        #trace2 = approx.sample(1000)
        return trace, inference, model, map

    if(method == 2):
        print('------- minibatch ADVI--------')
        minibatch_x = pm.Minibatch(X_train, batch_size=50)
        minibatch_y = pm.Minibatch(Y_train, batch_size=50)
        neural_network_minibatch = construct_nn(minibatch_x, minibatch_y,Xm, Y_train)
        with neural_network_minibatch as model:
            if(MAP == True):
                map = pm.find_MAP()
                inference = pm.ADVI()
                approx = pm.fit(n_, method=inference, start = map)
            else:
                map = pm.find_MAP()
                inference = pm.ADVI()
                approx = pm.fit(n_, method=inference)

        trace1 = approx.sample(1000)
        trace2 = approx.sample(1000)
        return trace1, trace2, inference, model, map

    if(method == 3):
        print('------- svgd--------')
        with Bayesian_Model as model:
            map = pm.find_MAP()
            approx = pm.fit(n_, start = map, method='svgd', inf_kwargs=dict(n_particles=100),
                         obj_optimizer=pm.sgd(learning_rate=0.01))




        return  approx, model, map

    if(method == 4):
        #n_chains = 3
        #test_folder = mkdtemp(prefix='SMC_TEST')
        print('------- Slice Sampling--------')
        with Bayesian_Model as model:
            map = pm.find_MAP()
            # instantiate sampler
            step = pm.Slice()
            # draw 5000 posterior samples
            trace = pm.sample(n_ , step=step, start=map, njobs=jobs)
        return trace, model, map

    if(method == 5):
        #n_chains = 2
        print('------- HamiltonianMC--------')
        with Bayesian_Model as model:
            map = pm.find_MAP()
            step = pm.HamiltonianMC()
            trace = pm.sample(n_ , start = map, chain= chains, njobs = jobs, step=step)
        return trace, model, map
    if(method == 6):
        #n_chains = 3
        print('------- Default--------')
        with Bayesian_Model as model:
            map = pm.find_MAP()
            trace = pm.sample(n_ , chain= chains, njobs=jobs,  callbacks=[CheckParametersConvergence()])
        return trace, model, map

    if(method == 7):
        #n_chains = 3
        print('------- Metropolis--------')
        with Bayesian_Model as model:
            # fmin_powell, optimize.basinhopping, fmin_BFGS, Newton-CG, 'trust-ncg', 'trust-krylov', 'trust-exact',
            #map = pm.find_MAP()
            map = 0
            #map = pm.find_MAP()
            step = pm.Metropolis()
            trace = pm.sample(n_ , step =step, chain= chains, njobs=jobs, callbacks=[CheckParametersConvergence()], tune =1000, step_size = 100)
            pm.traceplot(trace)
            name = 'Results/plots/'+star+'_'+classifier+'_PCA_'+str(PCA)+'2.png'
            plt.savefig(name)
            plt.clf()

        return trace, model, map

    if(method == 8):
        #n_chains = 3
        print('------- NUTS--------')
        with Bayesian_Model as model:
            # powell, BFGS, Newton-CG, 'trust-ncg', 'trust-krylov', 'trust-exact',
            map = pm.find_MAP(model = model, method = 'Newton-CG')
            step = pm.NUTS()
            trace = pm.sample(n_ ,step =step, start=map, chain= chains, njobs=jobs, callbacks=[CheckParametersConvergence()])
        return trace, model, map

    if(method == 9):
        #n_chains = 3
        print('------- SMC--------')
        with Bayesian_Model as model:
            # powell, BFGS, Newton-CG, 'trust-ncg', 'trust-krylov', 'trust-exact',
            map = pm.find_MAP(model = model, method = 'Newton-CG')
            step = pm.SMC()
            trace = pm.sample(n_ ,step =step, chain= chains, njobs=jobs, callbacks=[CheckParametersConvergence()])
            #print(model.marginal_likelihood)
        return trace, model, map



def predictionNN(X_test, neural_network, ann_input, X_train, approx):
    x = T.matrix('X')
    n = T.iscalar('n')
    x.tag.test_value = np.empty_like(X_train[:10])
    n.tag.test_value = 100
    _sample_proba = approx.sample_node(neural_network.out.distribution.p, size=n, more_replacements={ann_input: x})
    sample_proba = theano.function([x, n], _sample_proba)
    pred = sample_proba(X_test, 500).mean(0) > 0.5
    return pred


def analysisLogisticRegression(var1 = 0, var2 = 3, range_components = [1, 40, 5], method = 4, size_Sample = 2000):
    time_ml_i = []
    for i in range(range_components[0], range_components[1], range_components[2]):
        print("Number of components: ", i)
        c_comp = i
        np_scaled = preprocessing.normalize(Data_train)
        df_normalized = pd.DataFrame(np_scaled)
        pca = PCA(n_components=c_comp)
        principalComponents = pca.fit_transform(df_normalized)
        Reduced_Df = pd.DataFrame(data = principalComponents, columns = ['PC'+str(i) for i in range(c_comp)])
        result = pd.concat([label, Reduced_Df], axis=1)
        model = LogisticRegressionBinomial(result, var_label1=var1, var_label2=var2)
        start_1 = timeit.default_timer()
        trace, model = fitBayesianModel(model, method=method, n_=size_Sample, MAP = True)
        stop_1 = timeit.default_timer()
        time_post = stop_1 - start_1
        start_2 = timeit.default_timer()
        logml_dict = Marginal_llk(trace, model=model, maxiter=100000)
        print('The Bridge Sampling Estimatation of Logml is %.5f'%(logml_dict['logml']))
        stop_2 = timeit.default_timer()
        time_ml = stop_2 - start_2
        RLB_waic  = pm.waic(trace, model)
        time_ml_i.append([i, time_post, time_ml, logml_dict['logml'], RLB_waic.WAIC])
    return time_ml_i
