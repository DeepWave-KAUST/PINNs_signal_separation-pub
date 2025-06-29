import numpy as np
import scipy as sp
import pylops
from scipy.ndimage import median_filter, gaussian_filter
from pylops.basicoperators import Diagonal, FirstDerivative

try:
    import cupy as cp

    cp_asarray = cp.asarray
    cp_asnumpy = cp.asnumpy
    np_floatconv = np.float32
    np_floatcconv = np.complex64
    mempool = cp.get_default_memory_pool()
except:
    cp = np
    cp_asarray = np.asarray
    cp_asnumpy = np.asarray
    np_floatconv = np.float64
    np_floatcconv = np.complex128


def analytic_local_slope(d, dt, x, v0, kv, at_t=True):
    """Analytical local slopes for hyperbolic events

    Calculates analytical slope for hyperbolic events with
    linearly increasing root-mean square velocity: :math:`v_{rms}(t_0) = v_0 + k_v*t_0`

    Parameters
    ----------
    d : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    dt : :obj:`float`
        Time sampling
    x : :obj:`np.ndarray`
        Offset axis
    v0 : :obj:`float`
        Initial velocity at :math:`t_0 = 0 s`
    kv : :obj:`float`
        Velocity gradient
    at_t : :obj:`bool`, optional
        Return slopes at ``(t,x)`` (``True``) or
        ``(t0,x)`` (``False``)

    Parameters
    ----------
    slope : :obj:`np.ndarray`
        Local slopes of size :math:`n_x \times n_t`

    """

    [nx, nt] = d.shape
    t = np.arange(nt) * dt 
    
    v_rms = v0 + kv * t
    
    slope = np.zeros((nt, nx))

    for it0 in range(int(nt)):
        for ix in range(nx):
            tapp = np.sqrt(t[it0]**2 + (x[ix]/v_rms[it0])**2)
            it = (np.floor(tapp/dt+1)).astype(int)

            if at_t:
                if it < nt:
                    slope[it, ix] = x[ix] / (tapp * (v_rms[it0])**2)
            else:
                slope[it0, ix] = x[ix] / (tapp * (v_rms[it0])**2)

    # Put 0s to max value
    slope[slope == 0.] = np.nanmax(slope)
    
    # Put first spatial sample to 0
    slope[:, 0] = 0.

    return slope




def structure_tensor(d, dz=1.0, dx=1.0, smooth=5, eps=0.0, dips=False):
    r"""Local slope estimation with structure tensor algorithm

    This is a slighly modified version of :func:`pylops.utils.signalprocessing.slope_estimate` where fifth-order
    derivatives are used for the gradient leading to more accurate derivatives.

    Parameters
    ----------
    d : :obj:`np.ndarray`
        Input dataset of size :math:`n_z \times n_x`
    dz : :obj:`float`, optional
        Sampling in :math:`z`-axis, :math:`\Delta z`
    dx : :obj:`float`, optional
        Sampling in :math:`x`-axis, :math:`\Delta x`
    smooth : :obj:`float` or :obj:`np.ndarray`, optional
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.
    eps : :obj:`float`, optional
        Regularization term. All slopes where
        :math:`|g_{zx}| < \epsilon \max_{(x, z)} \{|g_{zx}|, |g_{zz}|, |g_{xx}|\}`
        are set to zero. All anisotropies where :math:`\lambda_\text{max} < \epsilon`
        are also set to zero. See Notes. When using with small values of ``smooth``,
        start from a very small number (e.g. 1e-10) and start increasing by a power
        of 10 until results are satisfactory.

    Returns
    -------
    slopes : :obj:`np.ndarray`
        Estimated local slopes. The unit is that of
        :math:`\Delta z/\Delta x`.

    """
    slopes = np.zeros_like(d)
    anisos = np.zeros_like(d)

    # gz, gx = np.gradient(d, dz, dx)
    Gx = pylops.FirstDerivative(dims=d.shape, axis=1, sampling=dx, order=5, edge=True)
    Gz = pylops.FirstDerivative(dims=d.shape, axis=0, sampling=dz, order=5, edge=True)
    gz, gx = Gz @ d, Gx @ d

    gzz, gzx, gxx = gz * gz, gz * gx, gx * gx

    # smoothing
    gzz = gaussian_filter(gzz, sigma=smooth)
    gzx = gaussian_filter(gzx, sigma=smooth)
    gxx = gaussian_filter(gxx, sigma=smooth)

    gmax = max(gzz.max(), gxx.max(), np.abs(gzx).max())
    if gmax <= eps:
        return np.zeros_like(d), anisos

    gzz /= gmax
    gzx /= gmax
    gxx /= gmax

    lcommon1 = 0.5 * (gzz + gxx)
    lcommon2 = 0.5 * np.sqrt((gzz - gxx) ** 2 + 4 * gzx ** 2)
    l1 = lcommon1 + lcommon2
    l2 = lcommon1 - lcommon2

    regdata = l1 > eps
    anisos[regdata] = 1 - l2[regdata] / l1[regdata]

    regdata = np.abs(gzx) > eps
    slopes[regdata] = (l1 - gzz)[regdata] / gzx[regdata]

    return slopes, anisos