'''This submodule contains utilities for the theory module.'''
import warnings
import numpy as np
from scipy.special import eval_hermitenorm, gamma, comb, factorial, gammainc
from scipy.stats import norm, multivariate_normal
from .base_th import _prepare_lkc


def get_σ(cls):
    """Compute the variance of a field with an angular power spectrum 'cls'.

    Parameters
    ----------
    cls : np.array
        The angular power spectrum of the Gaussian field.
    
    Returns
    -------
    σ² : float
        The variance of the field.
    
    """
    cls = np.array(cls, dtype=float)
    ℓ = np.arange(cls.shape[0])
    return np.sum(cls * (2.*ℓ+1.) / (4.*np.pi))


def get_μ(cls):
    """Compute the first derivative of the covariance function at the origin for a Gaussian field 
    defined on the sphere with angular power spectrum 'cls', which are normalised to unit variance.

    Parameters
    ----------
    cls : np.array
        The angular power spectrum of the Gaussian field.
    
    Returns
    -------
    μ : float
        The first derivative of the covariance function of a field at the origin.
    
    """
    cls = np.array(cls, dtype=float)
    ℓ = np.arange(cls.shape[0])
    cls /= np.sum(cls * (2.*ℓ+1.) / (4.*np.pi))
    μ = np.sum(cls * (2.*ℓ+1.) * ℓ*(ℓ+1.) / (8.*np.pi))
    return μ

def get_C2(cls):
    """Compute the second derivative of the covariance function at the origin for a Gaussian field 
    defined on the sphere with angular power spectrum 'cls', which are normalised to unit variance.

    Parameters
    ----------
    cls : np.array
        The angular power spectrum of the Gaussian field.
    
    Returns
    -------
    C2 : float
        The second derivative of the covariance function of a field at the origin.
    
    """
    cls = np.array(cls, dtype=float)
    ℓ = np.arange(cls.shape[0])
    cls /= np.sum(cls * (2.*ℓ+1.) / (4.*np.pi))
    C2 = np.sum(cls * (2.*ℓ+1.) * (ℓ-1.)*ℓ*(ℓ+1.)*(ℓ+2.) / (32.*np.pi))
    return C2

def flag(kj, k):
    """Compute flag coefficients.

    Parameters
    ----------
    ki : int
        The upper index of the flag coefficient.
        
    k : int
        The lower index of the flag coefficient.
    
    Returns
    -------
    flag : float
        The value of the float coefficient.
    
    """
    assert np.all(kj>=k), 'The first argument must be larger than the second.'
    return comb(kj, k) * omega(kj) / (omega(k) * omega(kj-k))
    
def omega(j):
    """Compute the volume of the $j$—dimensional ball, $\omega_j$.

    Parameters
    ----------
    j : int
        The dimension of the ball.
    
    Returns
    -------
    omega : float
        The volume of the $j$—dimensional ball.
    
    """
    return np.pi**(j/2.) / gamma(j/2. +1.)


def rho(k, us):
    """Compute the density functions $\rho_k(us)$ for Gaussian fields needed for the Gaussian Kinematic Formula.

    Parameters
    ----------
    k : int
        The order of the density function.
        
    us : np.array
        The thresholds where the density function is evaluated.
    
    Returns
    -------
    rho : np.array
        The density function evaluated at the thresholds.
    
    """
    if k==0:
        return 1. - norm.cdf(us)
    else:
        return 1. / (2. * np.pi)**(k/2.) * norm.pdf(us) * eval_hermitenorm(k-1, us)
    
def rho_Chi2(k, dof, us):
    """Compute the density functions $\rho_{k, \Chi^2}(us)$ for $\Chi^2_k$ fields (with $k$ degrees of freedom), needed for the Gaussian Kinematic Formula.

    Parameters
    ----------
    k : int
        The order of the density function.
        
    dof : int
        The degrees of freedom of the $\Chi^2$.
        
    us : np.array
        The thresholds where the density function is evaluated.
    
    Returns
    -------
    rho : np.array
        The density function evaluated at the thresholds.
    
    """
    if k==0:
        return 1.- (us >= 0.) * gammainc(dof/2., us/2.)
    else:
        factor = (us >= 0.) * us**((dof-k)/2.) * np.exp(-us/2.) / ((2.*np.pi)**(k/2.) * gamma(dof/2.) * 2.**((dof-2)/2.))
        summ = np.zeros_like(us)
        for l in np.arange(0, ((k-1)//2)+1):
            for m in np.arange(0,k-1-2*l+1):
                summ += (dof>= k - m - 2*l) * comb(dof-1, k-1-m-2*l) * (-1)**(k-1+m+l) * factorial(k-1) / (factorial(m) * factorial(l) * 2**l ) * us**(m+l)
        return factor*summ
    
def LKC(j, us, mu, dim=None, lkc_ambient=None):
    """Compute the expected value of the Lipschitz–Killing Curvatures (LKC) of the excursion set for Gaussian Isotropic fields.

    Parameters
    ----------
    j : int
        The index of the LKC.
        
    us : np.array
        The thresholds where the LKC is evaluated.

    mu : float
        The value of the derivative of the covariance function at the origin for the field.
        
    dim : int, optional
        The dimension of the ambient manifold.
        
    lkc_ambient : np.array, optional
        An array of the Lipschitz–Killing Curvatures of the ambient manifold. Its lenght must be `dim+1` if `dim` is also given.
        
    Returns
    ----------
    LKC : np.array
        The expected value of the Lipschitz–Killing Curvatures at the thresholds.
    """
    dim, lkc_ambient = _prepare_lkc(dim, lkc_ambient)
    result = np.zeros_like(us)
    for k in np.arange(0,dim-j+1):
        result += flag(k+j, k) * rho(k, us) * lkc_ambient[k+j] * mu**(k/2.)
    return result
    
def LKC_Chi2(j, us, mu, dof=2, dim=None, lkc_ambient=None):
    """Compute the expected value of the Lipschitz–Killing Curvatures (LKC) of the excursion set  for $\Chi^2_2$ fields.

    Parameters
    ----------
    j : int
        The index of the LKC.
        
    us : np.array
        The thresholds where the LKC is evaluated.

    mu : float
        The value of the derivative of the covariance function at the origin for the original Gaussian fields.

    dof : int, optional
        Number of degrees of freedom of the $\Chi^2$ field (i.e., number of squared Gaussian distributions that are summed). 
        Default: 2.
        
    dim : int, optional
        The dimension of the ambient manifold.
        
    lkc_ambient : np.array, optional
        An array of the Lipschitz–Killing Curvatures of the ambient manifold. Its lenght must be `dim+1` if `dim` is also given.
        
    Returns
    ----------
    LKC : np.array
        The expected value of the Lipschitz–Killing Curvatures at the thresholds.
    """
    dim, lkc_ambient = _prepare_lkc(dim, lkc_ambient)
    result = np.zeros_like(us)
    for k in np.arange(0,dim-j+1):
        result += flag(k+j, k) * rho_Chi2(k, dof, us) * lkc_ambient[k+j] * mu**(k/2.)
    return result

def rho_WNG(k, us, sigma, skewness, kurtosis, order=2):
    """Compute the density functions $\\rho_k(us)$ for weakly non-Gaussian fields, needed for the computation of the Lipschitz–Killing Curvatures.
    They are given by the perturbative expansion of [Matsubara & Kuriki (2021)](https://arxiv.org/abs/2011.04954), Eq. (91), in powers of the standard deviation of the field, $\\sigma$.

    Parameters
    ----------
    k : int
        The order of the density function.

    us : np.array
        The thresholds where the density function is evaluated, in units of the standard deviation of the field.

    sigma : float
        The standard deviation of the field, which is the parameter of the expansion.

    skewness : np.array
        The skewness parameters of the field, $[S^{(0)}, S^{(1)}, S^{(2)}]$.

    kurtosis : np.array
        The kurtosis parameters of the field, $[K^{(0)}, K^{(1)}, K^{(2)}_1, K^{(2)}_2, K^{(3)}]$.

    order : int, optional
        The order of the expansion in $\\sigma$: `0` (Gaussian), `1` or `2`.
        Default: 2.

    Returns
    -------
    rho : np.array
        The density function evaluated at the thresholds.

    """
    S0, S1, S2 = skewness
    K0, K1, K21, K22, K3 = kurtosis
    H = lambda n: eval_hermitenorm(n, us) if n >= 0 else 0.     # The terms with n<0 always have a null coefficient
    correction = np.zeros_like(us)
    if order >= 1:
        correction += sigma * (S0/6. * H(k+2) + k/3. * S1 * H(k) + k*(k-1)/6. * S2 * H(k-2))
    if order >= 2:
        correction += sigma**2. * (S0**2./72. * H(k+5)
                                   + (K0/24. + k/18. * S0*S1) * H(k+3)
                                   + k * (K1/8. + (k-1)/36. * S0*S2 + (k-2)/18. * S1**2.) * H(k+1)
                                   + k * ((k-2)/16. * K21 + k/16. * K22 + (k-1)*(k-4)/18. * S1*S2) * H(k-1)
                                   + k*(k-1)*(k-2) * (K3/24. + (k-7)/72. * S2**2.) * H(k-3))
    return rho(k, us) + norm.pdf(us) / (2.*np.pi)**(k/2.) * correction

def LKC_WNG(j, us, mu, sigma, skewness, kurtosis, order=2, dim=None, lkc_ambient=None):
    """Compute the expected value of the Lipschitz–Killing Curvatures (LKC) of the excursion set for weakly non-Gaussian Isotropic fields.

    Parameters
    ----------
    j : int
        The index of the LKC.

    us : np.array
        The thresholds where the LKC is evaluated, in units of the standard deviation of the field.

    mu : float
        The value of the derivative of the covariance function at the origin for the field.

    sigma : float
        The standard deviation of the field, which is the parameter of the expansion.

    skewness : np.array
        The skewness parameters of the field, $[S^{(0)}, S^{(1)}, S^{(2)}]$.

    kurtosis : np.array
        The kurtosis parameters of the field, $[K^{(0)}, K^{(1)}, K^{(2)}_1, K^{(2)}_2, K^{(3)}]$.

    order : int, optional
        The order of the expansion in $\\sigma$: `0` (Gaussian), `1` or `2`.
        Default: 2.

    dim : int, optional
        The dimension of the ambient manifold.

    lkc_ambient : np.array, optional
        An array of the Lipschitz–Killing Curvatures of the ambient manifold. Its length must be `dim+1` if `dim` is also given.

    Returns
    -------
    LKC : np.array
        The expected value of the Lipschitz–Killing Curvatures at the thresholds.
    """
    dim, lkc_ambient = _prepare_lkc(dim, lkc_ambient)
    result = np.zeros_like(us)
    for k in np.arange(0,dim-j+1):
        result += flag(k+j, k) * rho_WNG(k, us, sigma, skewness, kurtosis, order=order) * lkc_ambient[k+j] * mu**(k/2.)
    return result


def egoe(dim, a, b):
    """Compute the expected value of the Gaussian Orthogonal Ensamble, needed for the maxima distribution of Gaussian Isotropic fields.
    Details can be found in Cheng and Schwartzman, 2015. This formula is only proven for `a>0`, although it has been hypothetised to work for `a<0` under some conditions.
    
    Parameters
    ----------
    sim : int
        The dimension of the space (1 ≤ dim ≤ 3)
        
    a : float
        Parameter `a` of the formula. 

    b : float
        Parameter `b` of the formula.

    Returns
    ----------
    egoe : float
        The expected value needed for the computation of the maxima distribution.
    """
    assert dim in [1, 2, 3], '`dim` must be between 1 and 3'
    if a < 0:
        a = a + 0.j
        b = b + 0.j
        warnings.warn('This formula is only proven for `a>=0`, although it has been hypothetised to work for `a<0` under some conditions. If the maxima distribution looks wrong, it could be because of this.')
    # assert a > 0, '`a` must be positive'
    if dim == 1:
        return np.sqrt(4.*a+2.) * np.exp(-a*b**2./(2.*a +1.)) / (4.*a) + b*np.sqrt(np.pi/(2.*a)) * norm.cdf(b * np.sqrt(2.*a / (2.*a + 1)) )
    if dim == 2:
        return ( (1./a + 2.*b**2. - 1.) / np.sqrt(2.*a) * norm.cdf(b * np.sqrt(2.*a / (a + 1.))) +
            np.sqrt((a + 1.) / (2.*np.pi)) * np.exp(-a*b**2./(a +1.)) * b / a +
            np.sqrt(2./(2.*a+1.)) * np.exp(-a*b**2./(2.*a +1.)) * norm.cdf(a * b * np.sqrt(2. / ((2.*a + 1.)*(a + 1.))) ) )
    if dim == 3:
        Sigma1 = np.array([[1.5, -0.5], [-0.5, (1.+a)/(2.*a)]])
        Sigma2 = np.array([[1.5, -1.], [-1., (1.+2.*a)/(2.*a)]])
        term1 = (((24.*a**3. + 12.*a**2. + 6.*a + 1.) * b**2. / (2.*a * (2.*a+1)**2. ) +
                (6.*a**2 + 3.*a +2.) / (4.*a**2. * (2.*a+1)) + 3./2.) / np.sqrt(2. * (2.*a +1.)) * 
                np.exp(-a*b**2./(2.*a +1.)) * norm.cdf(2. * a * b * np.sqrt(2. / ((2.*a + 1.) * (2.*a +3.)) ) ) )
        term2 = (((a+1.) * b**2. / (2.*a) + (1.-a) / (2.*a**2.) - 1.) / np.sqrt(2. * (a +1.)) *
                np.exp(-a*b**2./(a +1.)) * norm.cdf(a * b * np.sqrt(2. / ((a + 1.) * (2.*a +3.)) ))  )
        term3 = (6.*a + 1. + (28.*a**2. + 12.*a + 3.) / (2.*a * (2.*a+1)) * 
                b / (2. * np.sqrt(2.*np.pi) * (2.*a+1.) * np.sqrt(2.*a +3.)) * np.exp(-3.*a*b**2./(2.*a +3.))  )
        term4 = (b**2. + 3. * (1.-a) / (2.*a)) * np.sqrt(np.pi/2.) * b/a * (multivariate_normal.cdf([0,b], cov=Sigma1) + multivariate_normal.cdf([0,b], cov=Sigma2))
        total = term1 + term2 + term3 + term4
        return total
    

lkc_ambient_dict = {"2D":np.array([0., 0., 1.]),
                    "3D":np.array([0., 0., 0., 1.]),
                    "sphere":np.array([0., 0., 4.*np.pi]),  # / (4.*np.pi),
                    "SO3":np.array([0., 6.*np.pi, 0., 4.*np.pi**2])}  # / (4.*np.pi**2)}
"""Dictionary with the characterization of different spaces through their Lipschitz–Killing Curvatures"""


__all__ = ["get_μ", "flag", "omega", "rho", "rho_Chi2", "rho_WNG", "LKC", "LKC_Chi2", "LKC_WNG", "lkc_ambient_dict"]

__docformat__ = "numpy"

 
