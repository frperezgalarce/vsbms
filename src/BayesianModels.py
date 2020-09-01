
# coding: utf-8

# In[ ]:

import theano
floatX = theano.config.floatX
theano.config.floatX = 'float64'
theano.config.compute_test_value ='raise'
theano.config.exception_verbosity = 'high'
import pymc3 as pm
import theano.tensor as T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.decomposition import PCA
from pymc3.theanof import set_tt_rng, MRG_RandomStreams
set_tt_rng(MRG_RandomStreams(42))
from sklearn import preprocessing
from pymc3.variational.callbacks import CheckParametersConvergence
import timeit
import sys
sys.setrecursionlimit(5000)
from scipy.linalg import lstsq
from statsmodels.tsa.ar_model import AR

def LogisticRegressionBinomialPrior(DataTrain, var_label1=0, var_label2 = 1, biasedSplit = False, priors = 'normal', onetoOne = True, className = 'Class', PCA = True,**kwargs):
    DataTrain['label']  = 1*(DataTrain[className] == 'ClassA')
    Y_train = DataTrain['label']
    del DataTrain[className]
    if PCA == True:
        List = ['PC'+str(i) for i in range(1, DataTrain.shape[1])]
    else:
        List = list(DataTrain.columns)
        try:
            List.remove('label')
        except:
            print('__')
    myList = '+'.join(map(str, List))
    label = 'label~'
    function = ''.join((label, myList))
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
    print('chains: ', chains)
    print('jobs: ', jobs)
    if(method == 1):
        print('-------SGFS--------')
        with Bayesian_Model as model:
            if(MAP==True):
                map = pm.find_MAP()
                inference = pm.SGFS(batch_size = 50, total_size = len(yTrain))
                trace = pm.sample(start = map, draws = n_, step = inference)
            else:
                map = pm.find_MAP()
                inference = pm.SGFS(batch_size = 50, total_size = len(yTrain))
                trace = pm.sample(draws = n_, step = inference)
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
        print('------- Slice Sampling--------')
        with Bayesian_Model as model:
            map = 0#pm.find_MAP()
            step = pm.Slice()
            trace = pm.sample(n_ , step=step, njobs=jobs)
        return trace, model, map

    if(method == 5):
        print('------- HamiltonianMC--------')
        with Bayesian_Model as model:
            #map = pm.find_MAP()
            step = pm.HamiltonianMC()
            trace = pm.sample(n_ , chain= chains, tune = 2000, njobs = jobs, step=step, init=None)
        return trace, model, map
    if(method == 6):
        print('------- Default--------')
        with Bayesian_Model as model:
            map = 0#pm.find_MAP()
            trace = pm.sample(n_ , chain= chains, njobs=jobs,  callbacks=[CheckParametersConvergence()])
        return trace, model, map

    if(method == 7):
        print('------- Metropolis--------')
        with Bayesian_Model as model:
            map = 0
            step = pm.Metropolis()
            trace = pm.sample(n_ , step =step, chain= chains, njobs=jobs, callbacks=[CheckParametersConvergence()], tune =1000, step_size = 100)
            pm.traceplot(trace)
            name = 'plots/'+star+'_'+classifier+'_PCA_'+str(PCA)+'2.png'
            plt.savefig(name)
            plt.clf()

        return trace, model, map

    if(method == 8):
        print('------- NUTS--------')

        with Bayesian_Model as model:
            stds = np.ones(model.ndim)
            for _ in range(5):
                args = {'is_cov': True}
                trace = pm.sample(500, tune=1000,chains = 1, init='advi+adapt_diag_grad', nuts_kwargs=args)
                samples = [model.dict_to_array(p) for p in trace]
                stds = np.array(samples).std(axis=0)
            traces = []
            for i in range(1):
                step = pm.NUTS(scaling=stds ** 2, is_cov=True, target_accept=0.8) #
                start = trace[-10 * i]

                trace_ = pm.sample(n_ , cores = 4, step=step, tune=1000, chain= chains, njobs=1, init='advi+adapt_diag_grad',start=start, callbacks=[CheckParametersConvergence()])
                #traces.append(trace_)
            trace = trace_#pm.backends.base.merge_traces(traces) #
            map = 0
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


def spectrum0_ar(x):
    """Port of spectrum0.ar from coda::spectrum0.ar"""
    z = np.arange(1, len(x)+1)
    z = z[:, np.newaxis]**[0, 1]
    p, res, rnk, s = lstsq(z, x)
    residuals = x - np.matmul(z, p)

    if residuals.std() == 0:
        spec = order = 0
    else:
        ar_out = AR(x).fit(ic='aic', trend='c')
        order = ar_out.k_ar
        spec = np.var(ar_out.resid)/(1 - np.sum(ar_out.params[1:]))**2

    return spec, order

def error_measures(logml):
    """Port of the error_measures.R in bridgesampling
    https://github.com/quentingronau/bridgesampling/blob/master/R/error_measures.R
    """
    ml = np.exp(logml['logml'])
    g_p = np.exp(logml['q12'])
    g_g = np.exp(logml['q22'])
    priorTimesLik_p = np.exp(logml['q11'])
    priorTimesLik_g = np.exp(logml['q21'])
    p_p = priorTimesLik_p/ml
    p_g = priorTimesLik_g/ml

    N1 = len(p_p)
    N2 = len(g_g)
    s1 = N1/(N1 + N2)
    s2 = N2/(N1 + N2)

    f1 = p_g/(s1*p_g + s2*g_g)
    f2 = g_p/(s1*p_p + s2*g_p)
    rho_f2, _ = spectrum0_ar(f2)

    term1 = 1/N2 * np.var( f1 ) / np.mean( f1 )**2
    term2 = rho_f2/N1 * np.var( f2 ) / np.mean( f2 )**2

    re2 = term1 + term2

    # convert to coefficient of variation (assumes that bridge estimate is unbiased)
    cv = np.sqrt(re2)
    print('The percentage errors of the estimation is: ', cv*100)
    return dict(re2 = re2, cv = cv)


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
