![Alt text](https://raw.githubusercontent.com/lpin0002/GammaBayes/main/misc/banner_logo_v2.png)


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


If you are using the GPU accelerated code please also cite the paper "How Can We Be Sure" which should be out soon.




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


If you use the default diffuse gamma-ray maps, they come from the GALPROP (https://arxiv.org/abs/2112.12745) package. 
If you use them please also cite them, and specifically the webrun (https://arxiv.org/abs/1008.3642, https://galprop.stanford.edu/webrun.php), which I used to get some of the results.

@article{Porter:2021tlr,
    author = "Porter, Troy A. and Johannesson, Gudlaugur and Moskalenko, V. Igor",
    title = "{The GALPROP Cosmic-ray Propagation and Nonthermal Emissions Framework: Release v57}",
    eprint = "2112.12745",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.3847/1538-4365/ac80f6",
    journal = "Astrophys. J. Supp.",
    volume = "262",
    number = "1",
    pages = "30",
    year = "2022"
}

@article{VLADIMIROV20111156,
title = {GALPROP WebRun: An internet-based service for calculating galactic cosmic ray propagation and associated photon emissions},
journal = {Computer Physics Communications},
volume = {182},
number = {5},
pages = {1156-1161},
year = {2011},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2011.01.017},
url = {https://www.sciencedirect.com/science/article/pii/S0010465511000373},
author = {A.E. Vladimirov and S.W. Digel and G. Jóhannesson and P.F. Michelson and I.V. Moskalenko and P.L. Nolan and E. Orlando and T.A. Porter and A.W. Strong},
keywords = {Astroparticle physics, Diffusion, Elementary particles, Cosmic rays, ISM: general, Dark matter, Diffuse radiation, Gamma rays: ISM, Infrared: ISM, Radio continuum: ISM, X-rays: ISM},
abstract = {GALPROP is a numerical code for calculating the galactic propagation of relativistic charged particles and the diffuse emissions produced during their propagation. The code incorporates as much realistic astrophysical input as possible together with latest theoretical developments and has become a de facto standard in astrophysics of cosmic rays. We present GALPROP WebRun, a service to the scientific community enabling easy use of the freely available GALPROP code via web browsers. In addition, we introduce the latest GALPROP version 54, available through this service.}
}




## Warning

Within the analysis we slice into matrices for the normalisation values of likelihood functions to enforce a normalisation on the interpolation done.
These matrices can be quite large depending on the resolution of the axes chosen. Keep this in mind when implementing multi-processing as python will
duplicate the arrays instead of reference the same one.

## Introduction

This coding repository contains a Bayesian Inference pipeline for calculating dark matter related observables from (simulated) observations from the galactic centre. Example files that run the simulation and analysis can be found within the `docs` folder. All documentation for the code is within the notebook files contained within that folder, that make up the [ReadTheDocs](https://gammabayes.readthedocs.io/en/latest/index.html) page and all the major components that make up the analysis in the oncoming publication.

GammaBayes 2.x requires **Python 3.10 or newer** and PyTorch >= 2.4 (CPU or CUDA). Install from
the repository with

```bash
pip install torch            # or the CUDA build for your driver, see https://pytorch.org
pip install -e src/          # from the repository root; add [docs,test] for the optional extras
```

A (0.x) release also exists on `PyPI` (`pip install gammabayes`); it predates the torch port and
does not match the current documentation. The dependencies are listed in `src/setup.py`
(note the `regions < 0.11` pin required by `gammapy <= 2.1`).

We recommend to look through the tutorials for an overview of the functionality of the code however, the figure below shows the main classes of `GammaBayes` in a UML diagram.


![Alt text](https://raw.githubusercontent.com/lpin0002/GammaBayes/main/misc/GammaBayes_UML.png)

