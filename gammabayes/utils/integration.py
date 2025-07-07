import numpy as np
from scipy import special
from astropy import units as u



def construct_log_dx(axis: np.ndarray) -> np.ndarray|float:
    """
    Constructs the logarithmic linear or log differences for the given axis.

    Args:
        axis (np.ndarray): Axis values.

    Returns:
        np.ndarray | float: Logarithmic differences for the axis.
    """
    try:
        axis = axis.value
    except:
        pass


    dx = np.diff(axis)

    
    
    # Isclose calculate with ``absolute(a - b) <= (atol + rtol * absolute(b))''
    # With the default values being atol=1e-8 and rtol=1e-5. Need to be careful
    # If we ever have differences ~1e-8 
    if np.isclose(dx[1], dx[0]):
        # Equal spacing, therefore the last dx can just be the same as the second last
        dx = np.append(dx, dx[-1])
    else:

        # Presuming log-uniform spacing
        dx = axis*(10**(np.log10(axis[1])-np.log10(axis[0])))

    return np.log(dx)

def construct_log_dx_mesh(axes: list[np.ndarray] | tuple[np.ndarray]) -> np.ndarray | float:
    """
    Constructs a logarithmic mesh of linear or log spaced axes.

    Args:
        axes (list[np.ndarray] | tuple[np.ndarray]): List or tuple of axis values.

    Returns:
        np.ndarray | float: Logarithmic differences mesh for the axes.
    """
    dxlist = []
    for axis in axes:
        dxlist.append(construct_log_dx(axis))

    logdx = np.sum(np.meshgrid(*dxlist, indexing='ij'), axis=0)

    return logdx


def logspace_riemann(logy: np.ndarray, x: np.ndarray, axis: int=-1) -> np.ndarray|float:
    """
    Performs Riemann sum integration in logarithmic integrand space for linear- or log-spaced axes.

    Args:
        logy (np.ndarray): Logarithm of function values.
        x (np.ndarray): Axis values.
        axis (int, optional): Axis along which to integrate. Defaults to -1.

    Returns:
        np.ndarray | float: Integrated result in logarithmic space.
    """

    logdx = construct_log_dx(x)
    indices = list(range(logy.ndim))
    indices.pop(axis)

    logdx = np.expand_dims(logdx, axis=indices)

    return special.logsumexp(logy+logdx, axis=axis)



def logspace_trapz(logy: np.ndarray, x: np.ndarray, axis: int =-1) -> np.ndarray|float:
    """
    Performs trapezoidal integration in logarithmic space.

    Args:
        logy (np.ndarray): Logarithm of function values.
        x (np.ndarray): Axis values.
        axis (int, optional): Axis along which to integrate. Defaults to -1.

    Returns:
        np.ndarray | float: Integrated result in logarithmic space.
    """
    h = np.diff(x)
    indices = list(range(logy.ndim))
    indices.pop(axis)

    h = np.expand_dims(h, axis=indices)


    # logfkm1    = logy[:-1]
    # logfk      = logy[1:]

    logfkm1 = np.take(logy, indices=np.arange(logy.shape[axis] - 1), axis=axis)
    logfk = np.take(logy, indices=np.arange(logy.shape[axis] - 1)+1, axis=axis)

    trapz_int = special.logsumexp(np.logaddexp(logfkm1, logfk)+np.log(h) - np.log(2), axis=axis)

    return trapz_int


def logspace_simpson(logy: np.ndarray, x: np.ndarray, axis: int =-1) -> np.ndarray|float:
    """
    Performs vectorized Simpson integration for log integrand values.

    Args:
        logy (np.ndarray): Logarithm of function values.
        x (np.ndarray): Axis values (evenly spaced in log10 space).
        axis (int, optional): Axis along which to integrate. Defaults to -1.

    Returns:
        np.ndarray | float: Integrated result using Simpson's rule in logarithmic space.
    """
    if logy.shape[axis] != len(x):
        raise ValueError("Input arrays y and x must have the same length.")
    
    logy = logy

    # Calculate the step size
    
    if len(x)>2:
        h = np.diff(x)
        length_h = len(h)
        # Sorry for naming convention I couldn't think of anything particularly better.
            # They refer to the indices that each array corresponds to when calculating
            # an integral when using the simpson algorithm
        logy_even_small = logy.take( indices=range(0,logy.shape[axis]-2,2), axis=axis )
        logy_odd        = logy.take( indices=range(1,logy.shape[axis]-1,2), axis=axis )
        logy_even_large = logy.take( indices=range(2,logy.shape[axis]-0,2), axis=axis )

        heven   = np.expand_dims(h[:-1:2],  axis=(*np.delete(np.arange(len(logy_even_small.shape)), axis),)) 
        hodd    = np.expand_dims(h[1::2],   axis=(*np.delete(np.arange(len(logy_even_small.shape)), axis),))

        term1 = np.log(2 - hodd/heven) + logy_even_small
        term2 = np.log((heven + hodd)**2 / (heven*hodd)) + logy_odd
        term3 = np.log(2 - heven/hodd) + logy_even_large

        result  = special.logsumexp(np.log(heven+hodd)+special.logsumexp([
                term1,
                term2, 
                term3
            ], axis=0), axis=axis)

        if length_h%2!=0:
            # Odd case, final interval 
            _alpha  = (2*h[-1]**2 + 3*h[-1]*h[-2])/(h[-2] + h[-1])
            _beta   = (h[-1]**2 + 3*h[-1]*h[-2])/h[-2]
            _eta    = h[-1]**3 / (h[-2] * (h[-2] + h[-1]))


            # To account for numerically unstable values we make them less unstable
                # by multiplying all the values such that the last entry in the 
                # relevant axis becomes 1
                # It isn't any guarantee though so be wary. Best solution if you're
                # worried it to simply have an even number of intervals 
                # i.e. length_h%2==0
            logy_m3     = logy.take( indices=-3, axis=axis )
            logy_m2     = logy.take( indices=-2, axis=axis )
            logy_m1     = logy.take( indices=-1, axis=axis )


            # We normally expect numerical underflow issues, not overflow
            final3logy = np.array([np.abs(logy_m1), np.abs(logy_m2), np.abs(logy_m3)])
            final3_noninf = final3logy[np.where(np.logical_not(np.isinf(final3logy)))]

            try:
                max_val = np.max(final3_noninf)/3
            except:
                # If they are all -inf then max_val doesn't matter
                max_val = 0


            addon = _alpha*np.exp(logy_m1+max_val) + _beta*np.exp(logy_m2+max_val) + -_eta*np.exp(logy_m3+max_val)

            if np.sum(addon<0)>0:
                result= np.log(np.exp(result+max_val) + addon) - max_val
            else:
                result= np.logaddexp(result, np.log(addon)-max_val)
        return result-np.log(6)
    else:
        # In the case where there are only three or less values
        return logspace_trapz(logy, x, axis=axis)


# Vectorised this function and chucked it into cython. 
    # best speed increase I got was about 6% with lots of loss in generality
    # Key bottlenecks if you can get around them is the creation of logdx
    # Also tried lintegrate library but couldn't get it to work
    # (from one of the issues doesn't seem to be 64-bit?)

def iterate_logspace_integration(logy: np.ndarray, axes: np.ndarray|tuple|list, logspace_integrator=logspace_riemann, axisindices: list=None) -> np.ndarray|float:
    """
    Iteratively integrates in integrand logspace over multiple axes.

    Args:
        logy (np.ndarray): Logarithm of function values.
        axes (np.ndarray): Axis values.
        logspace_integrator (callable, optional): Integration function to use. Defaults to logspace_riemann.
        axisindices (list, optional): List of axis indices to integrate over. Defaults to None.

    Raises:
        Exception: If an error occurs during integration.

    Returns:
        np.ndarray | float: Integrated result.
    """
            
    if axisindices is None:
        for axis in axes:
            logy = logspace_integrator(logy = logy, x=axis, axis=0)
    else:
        axisindices = np.asarray(axisindices)

        for loop_idx, (axis, axis_idx) in enumerate(zip(axes, axisindices)):

            # Assuming the indices are in order we subtract the loop idx from the axis index
            # print(loop_idx, axis_idx-loop_idx, axis.shape, logintegrandvalues.shape)
            try:
                logy = logspace_integrator(logy = logy, x=axis, axis=axis_idx-loop_idx)
            except Exception as excpt:
                print(loop_idx, axis.shape, axis_idx, logy.shape)
                raise Exception(f"Error occurred during integration --> {excpt}")



    return logy