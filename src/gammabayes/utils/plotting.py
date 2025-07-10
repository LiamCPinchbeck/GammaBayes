import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson as simps
import time
from matplotlib import cm
import sys, os
from scipy.special import logsumexp
from gammabayes.utils import logspace_simpson, logspace_riemann, iterate_logspace_integration
from gammabayes import bin_centres_to_edges, hdp_credible_interval_1d
from matplotlib.colors import LogNorm



defaults_kwargs = dict(
    smooth=1.0, 
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16), color='#0072C1',
    truth_color='tab:orange', 
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.), 1 - np.exp(-8), 1 - np.exp(-25/2)),
    plot_density=True, 
    plot_datapoints=True, 
    fill_contours=True,
    max_n_ticks=4, 
    )



def logdensity_matrix_plot(axes, log_dist_matrix, truevals=None, sigmalines_1d=True, sigmas=range(0,6), 
                           cmap=cm.get_cmap('Blues_r'), contours2d = False,
                           levels=np.array([1-np.exp(-25/2), 1-np.exp(-8), 1-np.exp(-4.5),1-np.exp(-2.0),1-np.exp(-0.5)]),
                           axis_names=None, suptitle='', suptitlesize=12, plot_density=False, norm='linear',
                           single_dim_ylabel='', vmin=None, vmax=None, iteratable_logspace_integrator=iterate_logspace_integration,
                           single_dim_logspace_integrator=logspace_riemann, axes_scale=None,
                           **kwargs):
    """
    Plots a matrix of log-density plots, including 1D and 2D marginal distributions.

    Args:
        axes (list[np.ndarray]): List of axis values.
        log_dist_matrix (np.ndarray): Logarithm of the distribution matrix.
        truevals (list[float], optional): True values to be marked on the plots. Defaults to None.
        sigmalines_1d (bool, optional): Whether to plot sigma lines on 1D plots. Defaults to True.
        sigmas (range, optional): Range of sigma values to plot. Defaults to range(0, 6).
        cmap (matplotlib.colors.Colormap, optional): Colormap for the plots. Defaults to cm.get_cmap('Blues_r').
        contours2d (bool, optional): Whether to plot 2D contours. Defaults to False.
        levels (np.ndarray, optional): Contour levels for 2D plots. Defaults to specific values.
        axis_names (list[str], optional): Names of the axes. Defaults to None.
        suptitle (str, optional): Suptitle for the entire plot. Defaults to ''.
        suptitlesize (int, optional): Font size of the suptitle. Defaults to 12.
        plot_density (bool, optional): Whether to plot density instead of raw values. Defaults to False.
        norm (str, optional): Scale for the y-axis. Defaults to 'linear'.
        single_dim_ylabel (str, optional): Y-label for single dimension plots. Defaults to ''.
        vmin (float, optional): Minimum value for color scaling. Defaults to None.
        vmax (float, optional): Maximum value for color scaling. Defaults to None.
        iteratable_logspace_integrator (callable, optional): Function for log-space integration over multiple axes. Defaults to iterate_logspace_integration.
        single_dim_logspace_integrator (callable, optional): Function for log-space integration over a single axis. Defaults to logspace_riemann.
        axes_scale (list[str], optional): Scale for each axis ('linear' or 'log'). Defaults to None.
        **kwargs: Additional keyword arguments for the plot.

    Returns:
        tuple: Figure and axes objects for the plot.
    """
    numaxes = len(axes)
    n = 1000

    fig, ax = plt.subplots(numaxes, numaxes, **kwargs)
    plt.suptitle(suptitle, size=suptitlesize)
    for rowidx in range(numaxes):
        for colidx in range(numaxes):
            if rowidx==colidx:
                integration_axes_indices = np.delete(np.arange(numaxes), [rowidx])
                integration_axes = [axes[integration_axis_idx] for integration_axis_idx in integration_axes_indices]
                log_marginal_dist = iteratable_logspace_integrator(log_dist_matrix, 
                                        axes=integration_axes, 
                                        axisindices=integration_axes_indices,)
                if plot_density:
                    marginal_dist = np.exp(log_marginal_dist-single_dim_logspace_integrator(logy=log_marginal_dist, x=axes[rowidx]))
                else:
                    marginal_dist = np.exp(log_marginal_dist)
                ax[rowidx,rowidx].plot(axes[rowidx], marginal_dist)
                if not(truevals is None):
                    ax[rowidx, rowidx].axvline(truevals[rowidx], ls='--', lw=1, c='tab:orange')
                    
                if norm!='log':
                    ax[rowidx,rowidx].set_ylim([0,None])
                ax[rowidx, rowidx].set_xlim([axes[colidx].min(), axes[colidx].max()])
                ax[rowidx,rowidx].set_yscale(norm)

                if axes_scale is not None:
                    ax[rowidx,rowidx].set_xscale(axes_scale[rowidx])



                if sigmalines_1d and plot_density:
                    print(marginal_dist.shape, axes[rowidx].shape)
                    for sigma in sigmas:
                        if sigma!=0:
                            color = cmap((np.abs(sigma)+0.1)/(2*len(sigmas)))
                        else:
                            color = 'tab:green'
                        for val in hdp_credible_interval_1d(y=marginal_dist, x=axes[rowidx], sigma=sigma):
                            ax[rowidx, rowidx].axvline(val, ls=':',c=color, alpha=0.8)

                if rowidx==0:
                    ax[rowidx, rowidx].set_ylabel(single_dim_ylabel)


                if not(axis_names is None) and rowidx==numaxes-1:
                    ax[rowidx, rowidx].set_xlabel(axis_names[rowidx])

            elif colidx<rowidx:
                integration_axes_indices = np.delete(np.arange(numaxes), [rowidx,colidx])
                integration_axes = [axes[integration_axis_idx] for integration_axis_idx in integration_axes_indices]
                log_marginal_dist = iteratable_logspace_integrator(log_dist_matrix, 
                                                axes=integration_axes, 
                                                axisindices=integration_axes_indices).T

                ax[rowidx, colidx].pcolormesh(axes[colidx], axes[rowidx], np.exp(log_marginal_dist), norm=norm, vmin=vmin, vmax=vmax)
                if truevals!=None:
                    ax[rowidx,colidx].axvline(truevals[colidx], ls='--', lw=1, c='tab:orange')
                    ax[rowidx,colidx].axhline(truevals[rowidx], ls='--', lw=1, c='tab:orange')

                ax[rowidx,colidx].set_xlim([axes[colidx].min(), axes[colidx].max()])
                ax[rowidx,colidx].set_ylim([axes[rowidx].min(), axes[rowidx].max()])
                if plot_density and contours2d:
                    normed_marginal_dist = np.exp(log_marginal_dist - logsumexp(log_marginal_dist))
                    t = np.linspace(0, normed_marginal_dist.max(), n)
                    integral = ((normed_marginal_dist >= t[:, None, None]) * normed_marginal_dist).sum(axis=(1,2))

                    f = interp1d(integral, t)
                    t_contours = f(levels)
                    ax[rowidx,colidx].contour(axes[colidx], axes[rowidx], normed_marginal_dist, levels=t_contours, colors='white', linewidths=0.5)
    
                if not(axis_names is None) and rowidx==numaxes-1:
                    ax[rowidx, colidx].set_xlabel(axis_names[colidx])
                if not(axis_names is None) and colidx==0:
                    ax[rowidx, colidx].set_ylabel(axis_names[rowidx])


                if axes_scale is not None:
                    ax[rowidx, colidx].set_xscale(axes_scale[rowidx])
                    ax[rowidx, colidx].set_xscale(axes_scale[colidx])

            else:
                ax[rowidx,colidx].set_axis_off()

    plt.tight_layout()

    return fig,ax
                


    