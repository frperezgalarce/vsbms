
# coding: utf-8

# In[2]:

import pymc3 as pm
import theano.tensor as T
import numpy as np
from warnings import filterwarnings, warn
filterwarnings('ignore')
from pymc3.theanof import set_tt_rng, MRG_RandomStreams
import itertools
set_tt_rng(MRG_RandomStreams(42))
from pymc3.model import modelcontext
from scipy import dot
from scipy.linalg import cholesky as chol
import scipy.stats as st
from tempfile import mkdtemp
from pymc3.step_methods import smc
from sklearn import preprocessing
from pymc3.variational.callbacks import CheckParametersConvergence
import timeit
import sys
from numpy import inf

sys.path.insert(0,'./src')
import sdpMatrix as sdp
# In[3]:

def Marginal_llk(mtrace, model=None, ADVI = False, trace2 = None, logp=None, maxiter=1000, burn_in=1000):
    """The Bridge Sampling Estimator of the Marginal Likelihood.
    Parameters
    ----------
    mtrace : MultiTrace, result of MCMC run
    model : PyMC Model Optional model. Default None, taken from context.
    logp : Model Log-probability function, read from the model by default
    maxiter : Maximum number of iterations
    Returns
    -------
    marg_llk : Estimated Marginal log-Likelihood.
    """
    r0, tol1, tol2 = 0.5, 1e-2, 1e-2

    model = modelcontext(model)
    if logp is None:
        logp = model.logp_array
    vars = model.free_RVs

    # Split the samples into two parts
    # Use the first 50% for fiting the proposal distribution and the second 50%
    # in the iterative scheme.
    #mtrace = mtrace[:,burn_in:]
    len_trace = len(mtrace)

    if ADVI == False:
        nchain = mtrace.nchains
        N1_ = len_trace // 2
        N1 = N1_*nchain
        N2 = len_trace*nchain - N1
        print('Number of chains: ', str(nchain))
        print('Number of samples: ',str(len_trace))
        #print(N1_)
        print('Samples to proposal distribution: ', str(N1))
        print('Samples to iterative scheme: ',str(N2))
        #print(pm.effective_n(mtrace[N1_:]))
        neff_list = dict() # effective sample size
    else:
        nchain = 2
        N1_ = len_trace
        N1 = N1_
        N2 =  len_trace

        print('Number of chains: ', str(nchain))
        print('Number of samples: ',str(len_trace))
        print('Samples to proposal distribution: ', str(N1))
        print('Samples to iterative scheme: ',str(N2))

    arraysz = model.bijection.ordering.size
    samples_4_fit = np.zeros((arraysz, N1))
    samples_4_iter = np.zeros((arraysz, N2))
    for var in vars:
        varmap = model.bijection.ordering.by_name[var.name]
        #print(varmap)
        neff_list = dict()
        # for fitting the proposal
        if ADVI == True:
            x = mtrace[0:N1_][var.name]
            samples_4_fit[varmap.slc, :] = x
        else:
            x = mtrace[0:N1_][var.name]
            samples_4_fit[varmap.slc, :] = x.reshape((x.shape[0],
                                                      np.prod(x.shape[1:], dtype=int))).T
        #print(x.shape)
        #print(x)

        # for the iterative scheme
        if ADVI == True:
            x2 = trace2[0:][var.name]
            samples_4_iter[varmap.slc, :] = x2
            # effective sample size of samples_4_iter, scalar
            neff_list.update(pm.effective_n(trace2[0:],varnames=[var.name]))

        else:
            x2 = mtrace[N1_:][var.name]
            samples_4_iter[varmap.slc, :] = x2.reshape((x2.shape[0],
                                                    np.prod(x2.shape[1:], dtype=int))).T
            # effective sample size of samples_4_iter, scalar
            neff_list.update(pm.effective_n(mtrace[N1_:],varnames=[var.name]))

    # median effective sample size (scalar)
    neff = pm.stats.dict2pd(neff_list,'temp').median()
    print(neff)
    # get mean & covariance matrix and generate samples from proposal
    m = np.mean(samples_4_fit, axis=1)
    print(m)
    V = np.cov(samples_4_fit)
    print(V)

    if np.all(np.linalg.eigvals(V)>0):
        L = chol(V, lower=True)
    else:
        print('SDP converting')
        V = sdp.nearPD(V)
        L = chol(V, lower=True)

    # Draw N2 samples from the proposal distribution
    print('m: ', np.sum(np.isinf(m[:, None])))


    #proposal bridge sampling
    gen_samples = m[:, None] + dot(L, st.norm.rvs(0, 1, size=samples_4_iter.shape))
    print('gen_samples: ', np.sum(np.isinf(gen_samples)))
    #gen_samples[gen_samples == inf] = 0
    # Evaluate proposal distribution for posterior & generated samples
    q12 = st.multivariate_normal.logpdf(samples_4_iter.T, m, V)
    q22 = st.multivariate_normal.logpdf(gen_samples.T, m, V)
    print('q12: ', np.sum(np.isinf(q12)))
    print('q22: ', np.sum(np.isinf(q22)))

    # Evaluate unnormalized posterior for posterior & generated samples
    q11 = np.asarray([logp(point) for point in samples_4_iter.T])
    q21 = np.asarray([logp(point) for point in gen_samples.T])

    print('q11: ', np.sum(np.isinf(q11)))
    print('q21: ', np.sum(np.isinf(q21)))
    #filterq21 = filter(lambda x: x != float('-inf'), q21)
    q21[np.isneginf(q21)] = -100000 #np.mean(q21[np.isfinite(q21)]) # replace inf by the mean -1755
    print('q21: ', np.sum(np.isinf(q21)))
    q11[np.isneginf(q11)] = -100000 #np.mean(q21[np.isfinite(q21)]) # replace inf by the mean -1755
    print('q11: ', np.sum(np.isinf(q11)))
    # Iterative scheme as proposed in Meng and Wong (1996) to estimate
    # the marginal likelihood
    def iterative_scheme(q11, q12, q21, q22, r0, neff, tol, maxiter, criterion):
        l1 = q11 - q12
        l2 = q21 - q22
        lstar =  np.median(l1) # To increase numerical stability,
                              # subtracting the median of l1 from l1 & l2 later

        print('neef: ', neff)
        s1 = neff/(neff + N2)
        s2 = N2/(neff + N2)
        r = r0
        r_vals = [r]
        logml = np.log(r) + lstar
        criterion_val = 1 + tol
        i = 0
        while (i <= maxiter) & (criterion_val > tol):
            print('i: ', i)
            print('maxiter', maxiter)
            print('criterionval: ', criterion_val)
            print('tol: ', tol)
            rold = r
            logmlold = logml


            numi = np.exp(l2 - lstar)/(s1 * np.exp(l2 - lstar) + s2 * r)
            print('l2: ', l2)
            print('lstar: ', lstar)
            print('s1: ', s1)
            print('r :', r)
            print('Num: ', numi)
            deni = 1/(s1 * np.exp(l1 - lstar) + s2 * r)
            print('Den: ', deni)


            if np.sum(~np.isfinite(numi))+np.sum(~np.isfinite(deni)) > 0:
                warn("""Infinite value in iterative scheme, returning NaN.
                Try rerunning with more samples.""")
            r = (N1/N2) * np.sum(numi)/np.sum(deni)
            print('r: ', r)
            r_vals.append(r)
            logml = np.log(r) + lstar
            print('Logml: ', logml)
            i += 1
            if criterion=='r':
                criterion_val = np.abs((r - rold)/r)
            elif criterion=='logml':
                criterion_val = np.abs((logml - logmlold)/logml)
            print('criterion val: ', criterion_val)

        if i >= maxiter:
            return dict(logml = np.NaN, niter = i, r_vals = np.asarray(r_vals))
        else:
            return dict(logml = logml, niter = i)

    # Run iterative scheme:
    tmp = iterative_scheme(q11, q12, q21, q22, r0, neff, tol1, maxiter, 'r')
    if ~np.isfinite(tmp['logml']):
        warn("""logml could not be estimated within maxiter, rerunning with
                      adjusted starting value. Estimate might be more variable than usual.""")
        # use geometric mean as starting value
        r0_2 = np.sqrt(tmp['r_vals'][-2]*tmp['r_vals'][-1])

        tmp = iterative_scheme(q11, q12, q21, q22, r0_2, neff, tol2, maxiter, 'r')

    return dict(logml = tmp['logml'], niter = tmp['niter'], method = "normal",
                q11 = q11, q12 = q12, q21 = q21, q22 = q22)


# In[ ]:
