![Alt text](https://raw.githubusercontent.com/lpin0002/GammaBayes/main/misc/banner_logo_v2.png)

__CODE CURRENTLY NOT IN OPERATIONAL STATE__
In the middle of a large code refactor and slowly integrating custom code that is not compatible with most of GammaBayes. If you still want to use GammaBayes I would recommend the use of the main branch (stable) or possibly developer branch (more stable than this branch).

__Author(s)__: Liam Pinchbeck (Liam.Pinchbeck@monash.edu)

__Supervisor(s)__: Csaba Balazs, Eric Thrane


__Documentation__: [ReadtheDocs](https://gammabayes.readthedocs.io/en/latest/index.html)

__Referencing__:

To reference this code please reference the following paper [2401.13876](https://arxiv.org/abs/2401.13876) or use the following bibtex.

@article{pinchbeck2024gammabayes,
      title={GammaBayes: a Bayesian pipeline for dark matter detection with CTA}, 
      author={Liam Pinchbeck and Eric Thrane and Csaba Balazs},
      year={2024},
      eprint={2401.13876},
      archivePrefix={arXiv},
      primaryClass={astro-ph.HE}
}

If you use the data from CosmiXs (https://arxiv.org/abs/2312.01153) or QCDUNC (referencing papers that have links to Zenodo databases;https://arxiv.org/abs/2303.11363, https://arxiv.org/abs/2202.11546 ,https://arxiv.org/abs/1812.07424) please also reference those sources. I have the BibTeX belo, but I'd double check them for you usages. And to be clear, we are not affiliated with these works, we just utilise their results.



@article{CosmiXs_ref,
  title   = {{CosmiXs: cosmic messenger spectra for indirect dark matter searches}},
  author  = {Arina, Chiara and Di Mauro, Mattia and Fornengo, Nicolao and
             Heisig, Jan and Jueid, Adil and Ruiz de Austri, Roberto},
  journal = {Journal of Cosmology and Astroparticle Physics},
  volume  = {2024},
  number  = {03},
  pages   = {035},
  year    = {2024},
  month   = mar,
  doi     = {10.1088/1475-7516/2024/03/035},
  url     = {https://doi.org/10.1088/1475-7516/2024/03/035}
}




@article{QCDUNC_Ref_1,
  title={{The Strong Force meets the Dark Sector: a robust estimate of QCD uncertainties for anti-matter dark matter searches}},
  author={Jueid, Adil and Kip, Jochem and Ruiz de Austri, Roberto and Skands, Peter},
  journal={Journal of High Energy Physics},
  volume={2024},
  number={2},
  pages={119},
  year={2024},
  publisher={Springer}
}

@article{QCDUNC_Ref_2,
   title={{Impact of QCD uncertainties on antiproton spectra from dark-matter annihilation}},
   volume={2023},
   ISSN={1475-7516},
   url={http://dx.doi.org/10.1088/1475-7516/2023/04/068},
   DOI={10.1088/1475-7516/2023/04/068},
   number={04},
   journal={Journal of Cosmology and Astroparticle Physics},
   publisher={IOP Publishing},
   author={Jueid, Adil and Kip, Jochem and Ruiz de Austri, Roberto and Skands, Peter},
   year={2023},
   month=apr, pages={068} }


@article{QCDUNC_Ref_3,
   title={{Estimating QCD uncertainties in Monte Carlo event generators for gamma-ray dark matter searches}},
   volume={2019},
   ISSN={1475-7516},
   url={http://dx.doi.org/10.1088/1475-7516/2019/05/007},
   DOI={10.1088/1475-7516/2019/05/007},
   number={05},
   journal={Journal of Cosmology and Astroparticle Physics},
   publisher={IOP Publishing},
   author={Amoroso, Simone and Caron, Sascha and Jueid, Adil and de Austri, Roberto Ruiz and Skands, Peter},
   year={2019},
   month=may, pages={007–007} }




## Warning

Within the analysis we slice into matrices for the normalisation values of likelihood functions to enforce a normalisation on the interpolation done.
These matrices can be quite large depending on the resolution of the axes chosen. Keep this in mind when implementing multi-processing as python will
duplicate the arrays instead of reference the same one.

## Introduction

This coding repository contains a Bayesian Inference pipeline for calculating dark matter related observables from (simulated) observations from the galactic centre. Example files that run the simulation and analysis can be found within the `docs` folder. All documentation for the code is within the notebook files contained within that folder, that make up the [ReadTheDocs](https://gammabayes.readthedocs.io/en/latest/index.html) page and all the major components that make up the analysis in the oncoming publication.

A stable python package version of the code exists on `PyPi` that can be installed with the command,

`pip install gammabayes`.

This will also take care of the required dependencies for the project. 

We recommend to look through the tutorials for an overview of the functionality of the code however, the figure below shows the main classes of `GammaBayes` in a UML diagram.


![Alt text](https://raw.githubusercontent.com/lpin0002/GammaBayes/main/misc/GammaBayes_UML.png)

