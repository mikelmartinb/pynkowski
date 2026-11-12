"""This submodule contains the definition for general Isotropic weakly non-Gaussian fields, whose statistics are computed with a perturbative expansion
around the Gaussian case.
"""
import numpy as np
from scipy.special import comb
from .base_th import TheoryField
from .utils_th import LKC_WNG


class WeaklyNonGaussian(TheoryField):
    """General class for Isotropic weakly non-Gaussian fields, such as the matter density field in the weakly non-linear regime.

    The expected values of the Minkowski Functionals are computed with the perturbative expansion of [Matsubara & Kuriki (2021)](https://arxiv.org/abs/2011.04954), Eq. (91),
    in powers of the standard deviation of the field, $\\sigma$, up to second order. At order 0, they are the ones of an Isotropic Gaussian field. The first order corrections
    depend on the skewness parameters, $S^{(a)}$, and the second order corrections also depend on the kurtosis parameters, $K^{(a)}$. They are defined (Eqs. 76 and 77)
    from the cumulants of the field $f$ and its derivatives, with $\\sigma_1^2 = \\langle |\\nabla f|^2 \\rangle$:

    - $S^{(0)} = \\langle f^3 \\rangle_c / \\sigma^4$,
    - $S^{(1)} = \\frac{3}{2} \\langle f |\\nabla f|^2 \\rangle_c / (\\sigma^2 \\sigma_1^2)$,
    - $S^{(2)} = -\\frac{3d}{2(d-1)} \\langle |\\nabla f|^2 \\nabla^2 f \\rangle_c / \\sigma_1^4$,
    - $K^{(0)} = \\langle f^4 \\rangle_c / \\sigma^6$,
    - $K^{(1)} = 2 \\langle f^2 |\\nabla f|^2 \\rangle_c / (\\sigma^4 \\sigma_1^2)$,
    - $K^{(2)}_1 = -\\frac{2d}{(d+2)(d-1)} \\left[(d+2) \\langle f |\\nabla f|^2 \\nabla^2 f \\rangle_c + \\langle |\\nabla f|^4 \\rangle_c\\right] / (\\sigma^2 \\sigma_1^4)$,
    - $K^{(2)}_2 = -\\frac{2d}{(d+2)(d-1)} \\left[(d+2) \\langle f |\\nabla f|^2 \\nabla^2 f \\rangle_c + d \\langle |\\nabla f|^4 \\rangle_c\\right] / (\\sigma^2 \\sigma_1^4)$,
    - $K^{(3)} = \\frac{2d^2}{(d-1)(d-2)} \\left[\\langle |\\nabla f|^2 (\\nabla^2 f)^2 \\rangle_c - \\langle |\\nabla f|^2 f_{;ij} f_{;ij} \\rangle_c\\right] / \\sigma_1^6$.

    $S^{(2)}$ is not used if `dim=1`, only $K^{(2)}_1 - K^{(2)}_2$ is used if `dim=1`, and $K^{(3)}$ is not used if `dim<3`. These parameters can be computed from a data field
    with `pynkowski.stats.minkowski.cumulants`. The corrections only depend on the combinations $\\sigma S^{(a)}$ and $\\sigma^2 K^{(a)}$, which do not change with the normalisation
    of the field: the parameters must be the ones of the field with standard deviation `sigma` (e.g., for a field normalised to unit variance, `sigma=1` and the parameters of the normalised field).
    The expansion is accurate when $\\sigma S^{(a)} \\ll 1$ and $\\sigma^2 K^{(a)} \\ll 1$. It is derived for Euclidean spaces; on curved spaces, such as the sphere, it is valid when the
    correlation length of the field is small compared to the curvature radius of the space.

    Parameters
    ----------
    dim : int
        Dimension of the space where the field is defined.

    sigma : float, optional
        The standard deviation of the field, which is also the parameter of the expansion.
        Default : 1.

    mu : float, optional
        The derivative of the covariance function at the origin, times $-2$. Equal to the variance of the first derivatives of the field normalised to unit variance, $\\sigma_1^2/(d\\sigma^2)$.
        Default : 1.

    skewness : np.array or None, optional
        The skewness parameters of the field, $[S^{(0)}, S^{(1)}, S^{(2)}]$. If `None`, they are 0.
        Default : None

    kurtosis : np.array or None, optional
        The kurtosis parameters of the field, $[K^{(0)}, K^{(1)}, K^{(2)}_1, K^{(2)}_2, K^{(3)}]$. They are only used if `order=2`. If `None`, they are 0.
        Default : None

    order : int, optional
        The order of the expansion in $\\sigma$: `0` (Gaussian field), `1` (corrections from the skewness parameters) or `2` (corrections from the skewness and kurtosis parameters).
        Default : 2

    lkc_ambient : np.array or None, optional
        The values for the Lipschitz–Killing Curvatures of the ambient space. If `None`, it is assumed that the volume is 1 and the rest is 0, as for `pynkowski.theory.gaussian.Gaussian`.
        Default : None

    Attributes
    ----------
    dim : int
        Dimension of the space where the field is defined.

    name : str
        Name of the field, `"Isotropic Weakly Non-Gaussian"` by default.

    sigma : float
        The standard deviation of the field.

    mu : float
        The derivative of the covariance function at the origin, times $-2$. Equal to the variance of the first derivatives of the field normalised to unit variance.

    skewness : np.array
        The skewness parameters of the field, $[S^{(0)}, S^{(1)}, S^{(2)}]$.

    kurtosis : np.array
        The kurtosis parameters of the field, $[K^{(0)}, K^{(1)}, K^{(2)}_1, K^{(2)}_2, K^{(3)}]$.

    order : int
        The order of the expansion in $\\sigma$.

    lkc_ambient : np.array or None
        The values for the Lipschitz–Killing Curvatures of the ambient space.

    """
    def __init__(self, dim, sigma=1., mu=1., skewness=None, kurtosis=None, order=2, lkc_ambient=None):
        super().__init__(dim, name='Isotropic Weakly Non-Gaussian', sigma=sigma, mu=mu, lkc_ambient=lkc_ambient)
        if order not in (0, 1, 2):
            raise ValueError(f"`order` must be 0, 1 or 2, not {order}.")
        self.order = order
        self.skewness = np.zeros(3) if skewness is None else np.asarray(skewness, dtype=float)
        self.kurtosis = np.zeros(5) if kurtosis is None else np.asarray(kurtosis, dtype=float)
        if self.skewness.shape != (3,) or self.kurtosis.shape != (5,):
            raise ValueError(f"`skewness` and `kurtosis` must have 3 and 5 elements, not {self.skewness.shape} and {self.kurtosis.shape}.")

    def LKC(self, j, us):
        """Compute the expected values of the Lipschitz–Killing Curvatures of the excursion sets at thresholds `us`, $\\mathbb{L}_j(A_u(f))$.

        Parameters
        ----------
        j : int
            Index of the LKC to compute, `0 ≤ j ≤ dim`.

        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        lkc : np.array
            Expected value of LKC evaluated at the thresholds.

        """
        us = us / self.sigma
        return LKC_WNG(j, us, self.mu, self.sigma, self.skewness, self.kurtosis, order=self.order, dim=self.dim, lkc_ambient=self.lkc_ambient)/self.lkc_ambient[-1]

    def MF(self, j, us):
        """Compute the expected values of the Minkowski Functionals of the excursion sets at thresholds `us`, $V_j(A_u(f))$.

        Parameters
        ----------
        j : int
            Index of the MF to compute, `0 ≤ j ≤ dim`.

        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        mf : np.array
            Expected value of MFs evaluated at the thresholds.

        """
        return 1./comb(self.dim,j)*self.LKC(self.dim-j, us)

    def V0(self, us):
        """Compute the expected values of the first Minkowski Functionals of the excursion sets at thresholds `us`, $V_0(A_u(f))$.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        v0 : np.array
            Expected value of $V_0$ evaluated at the thresholds.

        """
        return self.MF(0, us)

    def V1(self, us):
        """Compute the expected values of the second Minkowski Functionals of the excursion sets at thresholds `us`, $V_1(A_u(f))$.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        v1 : np.array
            Expected value of $V_1$ evaluated at the thresholds.

        """
        return self.MF(1, us)

    def V2(self, us):
        """Compute the expected values of the third Minkowski Functionals of the excursion sets at thresholds `us`, $V_2(A_u(f))$. Only for fields with `dim ≥ 2`.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        v2 : np.array
            Expected value of $V_2$ evaluated at the thresholds.

        """
        assert self.dim>=2, 'V2 is defined only for fields with dim≥2'
        return self.MF(2, us)

    def V3(self, us):
        """Compute the expected values of the fourth Minkowski Functionals of the excursion sets at thresholds `us`, $V_3(A_u(f))$. Only for fields with `dim ≥ 3`.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        v3 : np.array
            Expected value of $V_3$ evaluated at the thresholds.

        """
        assert self.dim>=3, 'V3 is defined only for fields with dim≥3'
        return self.MF(3, us)


__all__ = ["WeaklyNonGaussian"]

__docformat__ = "numpy"
