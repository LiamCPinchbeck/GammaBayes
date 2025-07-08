import numpy as np
from scipy import special
from astropy import units as u
import torch


def construct_log_dx(axis):
    """
    Constructs the logarithmic linear or log differences for the given axis.

    Args:
        axis: Axis values.

    Returns:
        Logarithmic differences for the axis.
    """
    try:
        axis = axis.value
    except:
        pass


    dx = torch.diff(axis)

    
    
    # Isclose calculate with ``absolute(a - b) <= (atol + rtol * absolute(b))''
    # With the default values being atol=1e-8 and rtol=1e-5. Need to be careful
    # If we ever have differences ~1e-8 
    if torch.isclose(dx[1], dx[0]):
        # Equal spacing, therefore the last dx can just be the same as the second last
        dx = torch.cat((dx, dx[-1]))
    else:

        # Presuming log-uniform spacing
        dx = axis*(10**(torch.log10(axis[1])-torch.log10(axis[0])))

    return torch.log(dx)

def construct_log_dx_mesh(axes):
    dxlist = []
    for axis in axes:
        dxlist.append(construct_log_dx(axis))

    logdx = torch.sum(torch.meshgrid(*dxlist, indexing='ij'), axis=0)

    return logdx


def logspace_riemann(logy, x, axis: int=-1):
    """
    Performs Riemann sum integration in logarithmic integrand space for linear- or log-spaced axes.

    Args:
        logy (): Logarithm of function values.
        x (): Axis values.
        axis (int, optional): Axis along which to integrate. Defaults to -1.

    Returns:
        Integrated result in logarithmic space.
    """

    logdx = construct_log_dx(x)
    indices = list(range(logy.ndim))
    indices.pop(axis)

    for index in indices:
        logdx = logdx.unsqueeze(index)

    return special.logsumexp(logy+logdx, axis=axis)




def iterate_logspace_integration(logy, axes, logspace_integrator=logspace_riemann, axisindices: list=None):
    """
    Iteratively integrates in integrand logspace over multiple axes.

    Args:
        logy: Logarithm of function values.
        axes: Axis values.
        logspace_integrator (callable, optional): Integration function to use. Defaults to logspace_riemann.
        axisindices (list, optional): List of axis indices to integrate over. Defaults to None.

    Raises:
        Exception: If an error occurs during integration.

    Returns:
        Integrated result.
    """
            
    if axisindices is None:
        for axis in axes:
            logy = logspace_integrator(logy = logy, x=axis, axis=0)
    else:
        axisindices = torch.tensor(axisindices)

        for loop_idx, (axis, axis_idx) in enumerate(zip(axes, axisindices)):

            # Assuming the indices are in order we subtract the loop idx from the axis index
            # print(loop_idx, axis_idx-loop_idx, axis.shape, logintegrandvalues.shape)
            try:
                logy = logspace_integrator(logy = logy, x=axis, axis=axis_idx-loop_idx)
            except Exception as excpt:
                print(loop_idx, axis.shape, axis_idx, logy.shape)
                raise Exception(f"Error occurred during integration --> {excpt}")



    return logy