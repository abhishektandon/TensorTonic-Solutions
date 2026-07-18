import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    pmf = (np.exp(-lam) * lam ** k)/np.prod(np.arange(1, k+1))
    cmf = np.sum([(np.exp(-lam) * lam ** i)/np.prod(np.arange(1, i+1)) for i in range(k+1)])

    return pmf, cmf