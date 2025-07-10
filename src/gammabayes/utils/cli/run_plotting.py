from dynesty import NestedSampler
import yaml, sys, os, time, numpy as np
from gammabayes.high_level_inference import High_Level_Analysis
from corner import corner
from gammabayes.utils.plotting import defaults_kwargs
from scipy.special import logsumexp, erf
from matplotlib import pyplot as plt
from gammabayes import Parameter
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc






def plot_from_save(analysis_class=None, samples=None, sampler_results = None, save_str="", diag_str="", save_dir="Posteriors"):

    if analysis_class is None or samples is None:
        with open(sys.argv[1], 'r') as file:
            config_file = yaml.safe_load(file)

        analysis_class = High_Level_Analysis.from_config_dict(config_file, skip_irf_norm=True)

        sampler = NestedSampler.restore(analysis_class.config_dict['dynesty_kwargs']['run_kwargs']['checkpoint_file'])


        sampler_results = sampler.results

        samples = sampler_results.samples

         

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"data/{analysis_class.config_dict['stem_identifier']}/diagnostic_plots", exist_ok=True)
    stddevvals = np.array([-5, -4, 3, -2, -1, 1, 2, 3, 4, 5])
    quantiles = 0.5*(1+erf(stddevvals/np.sqrt(2)))

    titlestddvals = np.array([-2, 0, 2,])
    title_quantiles = 0.5*(1+erf(titlestddvals/np.sqrt(2)))

    titlestddvals = np.array([-2, 0, 2,])
    title_quantiles = 0.5*(1+erf(titlestddvals/np.sqrt(2)))

    defaults_kwargs['smooth'] = 3.0
    # del defaults_kwargs['label_kwargs']['fontsize']
    # del defaults_kwargs['title_kwargs']['fontsize']

    defaults_kwargs['label_kwargs'] = {"rotation": 45, 'labelpad':25, 'size':16}


    scales = []
    for mixture_param in analysis_class.mixture_parameter_specifications.values():
        mixture_param = Parameter(mixture_param)
        if mixture_param.scaling!='log10':
            scales.append('linear')
        else:
            scales.append('log')

    if analysis_class.obs_prior_parameter_specifications[0].dict_of_parameters_by_name['mass'].scaling!='log10':
        scales.append(analysis_class.obs_prior_parameter_specifications[0].dict_of_parameters_by_name['mass'].scaling)
    else:
        scales.append('log')

    if sampler_results is None:
        weights = None
    else:
        weights = np.exp(sampler_results.logwt-np.max(sampler_results.logwt))

    if not("equal" in save_str):
                title_quantiles=None
                quantiles=None

    ranges = [[sampling_results_sample_set.min(), sampling_results_sample_set.max()] for sampling_results_sample_set in samples.T]


    # high_level_mixture_frac_labels = list(analysis_class.config_dict['true_mixture_fractions'].keys())
    # true_mixture_fractions_values = list(analysis_class.config_dict['true_mixture_fractions'].values())
    # dm_level_mixture_frac_labels = analysis_class.DM_Annihilation_Ratios.keys()

    # truths=[true_mixture_fractions_values[0], *analysis_class.DM_Annihilation_Ratios.values(), *true_mixture_fractions_values[1:], analysis_class.config_dict['dark_matter_mass']]
    # fig = plt.figure(figsize=(12,12), dpi=100)
    #mixture_parameter_set
    figure=corner(samples, 
                # fig=fig,
                weights = weights,
                title_fmt='.6f',
                # truths=truths, 
                show_titles=True,
                bins = 64,
                axes_scale= scales,
                range = ranges,
                title_quantiles=title_quantiles,
                quantiles=quantiles,
                # labels=[high_level_mixture_frac_labels[0], *dm_level_mixture_frac_labels, *high_level_mixture_frac_labels[1:], 'mass [TeV]'],
                **defaults_kwargs,
                dpi=300)
    plt.suptitle(f"Events = {analysis_class.config_dict['Nevents']}, neff = {len(samples[:, 0])}", size=18)

    # Adjust the position of the labels
    for ax in figure.get_axes():
        if ax.get_xlabel():
            label = ax.get_xlabel()
            ax.set_xlabel('')  # Remove original label
            ax.annotate(label, xy=(0.5, -0.5), xycoords='axes fraction', ha='center', va='top', rotation=45, size=12)  # Add new label with desired padding

        if ax.get_ylabel():
            label = ax.get_ylabel()
            ax.set_ylabel('')  # Remove original label
            ax.annotate(label, xy=(-0.5, 0.5), xycoords='axes fraction', ha='right', va='center', rotation=45, size=12)  # Add new label with desired padding

    plt.tight_layout()




    plt.savefig(time.strftime(f"{save_dir}/{analysis_class.config_dict['stem_identifier']}_{save_str}_posterior_%m|%d_%H:%M.pdf"))
    plt.savefig(time.strftime(f"data/{analysis_class.config_dict['stem_identifier']}/diagnostic_plots/{analysis_class.config_dict['stem_identifier']}_{save_str}_posterior_%m|%d_%H:%M.pdf"))

    plt.close()

    plt.tight_layout()
    np.save(time.strftime(f'data/{analysis_class.config_dict['stem_identifier']}/{analysis_class.config_dict['stem_identifier']}_{save_str}__posterior_%m|%d_%H'), samples)


    if not(sampler_results is None):


        fig, axes = dyplot.runplot(sampler_results, 
                                   logplot=True)
        plt.tight_layout()
        fig.savefig(time.strftime(f"data/{analysis_class.config_dict['stem_identifier']}/diagnostic_plots/{analysis_class.config_dict['stem_identifier']}_{diag_str}_summary_diagnostic_%m|%d_%H:%M.pdf"))
        plt.close()


        fig, axes = dyplot.traceplot(sampler_results, 
                                # truths=truths,
                                truth_color='tab:orange', 
                                trace_cmap='viridis',)
        
        plt.tight_layout()
        
        fig.savefig(time.strftime(f"data/{analysis_class.config_dict['stem_identifier']}/diagnostic_plots/{analysis_class.config_dict['stem_identifier']}_{diag_str}_trace_diagnostic_%m|%d_%H:%M.pdf"))
        plt.close()








if __name__=="__main__":
    with open(sys.argv[1], 'r') as file:
        config_file = yaml.safe_load(file)

    analysis_class = High_Level_Analysis.from_config_dict(config_file, skip_irf_norm=True)

    sampler = NestedSampler.restore(analysis_class.config_dict['dynesty_kwargs']['run_kwargs']['checkpoint_file'])


    sampler_results = sampler.results

    equal_samples = sampler_results.samples_equal()

    raw_samples = sampler_results.samples


    plot_from_save(analysis_class, raw_samples, sampler_results=sampler_results, save_str="raw")
    plot_from_save(analysis_class, equal_samples, save_str="equal")






