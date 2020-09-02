import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

from pymc3.variational.callbacks import CheckParametersConvergence
from scipy.linalg import lstsq
from statsmodels.tsa.ar_model import AR
from warnings import filterwarnings

filterwarnings('ignore')


def fitbayesianmodel(bayesian_model, ytrain, method=1, n_=3000, MAP=True, chains=1, jobs=1, star='rrlyr',
                     classifier='RL', PCA=False):
    print('chains: ', chains)
    print('jobs: ', jobs)
    if method == 4:
        print('------- Slice Sampling--------')
        with bayesian_model as model:
            map = 0
            step = pm.Slice()
            trace = pm.sample(n_, step=step, njobs=jobs)
        return trace, model, map

    if method == 5:
        print('------- HamiltonianMC--------')
        with bayesian_model as model:
            step = pm.HamiltonianMC()
            trace = pm.sample(n_, chain=chains, tune=2000, njobs=jobs, step=step, init=None)
        return trace, model, map
    if method == 6:
        print('------- Default--------')
        with bayesian_model as model:
            map = 0
            trace = pm.sample(n_, chain=chains, njobs=jobs, callbacks=[CheckParametersConvergence()])
        return trace, model, map

    if method == 7:
        print('------- Metropolis--------')
        with bayesian_model as model:
            map = 0
            step = pm.Metropolis()
            trace = pm.sample(n_, step=step, chain=chains, njobs=jobs, callbacks=[CheckParametersConvergence()],
                              tune=1000, step_size=100)
            pm.traceplot(trace)
            name = star + '_' + classifier + '_PCA_' + str(PCA) + '2.png'
            plt.savefig(name)
            plt.clf()

        return trace, model, map

    if method == 8:
        print('------- NUTS--------')

        with bayesian_model as model:
            stds = np.ones(model.ndim)
            for _ in range(5):
                args = {'is_cov': True}
                trace = pm.sample(500, tune=1000, chains=1, init='advi+adapt_diag_grad', nuts_kwargs=args)
                samples = [model.dict_to_array(p) for p in trace]
                stds = np.array(samples).std(axis=0)
            traces = []
            for i in range(1):
                step = pm.NUTS(scaling=stds ** 2, is_cov=True, target_accept=0.8)  #
                start = trace[-10 * i]

                trace_ = pm.sample(n_, cores=4, step=step, tune=1000, chain=chains, njobs=1,
                                   init='advi+adapt_diag_grad', start=start, callbacks=[CheckParametersConvergence()])
            trace = trace_
            map = 0
        return trace, model, map


def spectrum0_ar(x):
    z = np.arange(1, len(x) + 1)
    z = z[:, np.newaxis] ** [0, 1]
    p, res, rnk, s = lstsq(z, x)
    residuals = x - np.matmul(z, p)

    if residuals.std() == 0:
        spec = order = 0
    else:
        ar_out = AR(x).fit(ic='aic', trend='c')
        order = ar_out.k_ar
        spec = np.var(ar_out.resid) / (1 - np.sum(ar_out.params[1:])) ** 2

    return spec, order


def error_measures(logml):
    ml = np.exp(logml['logml'])
    g_p = np.exp(logml['q12'])
    g_g = np.exp(logml['q22'])
    priorTimesLik_p = np.exp(logml['q11'])
    priorTimesLik_g = np.exp(logml['q21'])
    p_p = priorTimesLik_p / ml
    p_g = priorTimesLik_g / ml
    N1 = len(p_p)
    N2 = len(g_g)
    s1 = N1 / (N1 + N2)
    s2 = N2 / (N1 + N2)
    f1 = p_g / (s1 * p_g + s2 * g_g)
    f2 = g_p / (s1 * p_p + s2 * g_p)
    rho_f2, _ = spectrum0_ar(f2)
    term1 = 1 / N2 * np.var(f1) / np.mean(f1) ** 2
    term2 = rho_f2 / N1 * np.var(f2) / np.mean(f2) ** 2
    re2 = term1 + term2
    cv = np.sqrt(re2)
    print('The percentage errors of the estimation is: ', cv * 100)
    return dict(re2=re2, cv=cv)
