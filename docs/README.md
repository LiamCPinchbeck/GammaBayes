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

A stable python package version of the code exists on `PyPi` that can be installed with the command,

`pip install gammabayes`.

This will also take care of the required dependencies for the project. 

We recommend to look through the tutorials for an overview of the functionality of the code however, the figure below shows the main classes of `GammaBayes` in a UML diagram.


![Alt text](https://raw.githubusercontent.com/lpin0002/GammaBayes/main/misc/GammaBayes_UML.png)







# Building the GammaBayes documentation

```
docs/
  conf.py, index.rst          Sphinx configuration and front page
  getting_started/            installation, concepts, migration notes (markdown)
  tutorials/*.ipynb           executable tutorials for the 2.x torch API
  api/                        API reference (autodoc)
  legacy_tutorials/           the v0.x notebooks (not built; kept for reference)
  run_notebooks.sh            executes the tutorials in place (needs the package + IRF files)
  annotate_timings.py         stamps executed notebooks with per-cell wall times (called by the script)
```

> **Requires GammaBayes >= 2.1.1**: the tutorials use `GammaBinning.padded`,
> `IRFExtractor(irf_time_seconds=...)`, `IsotropicPowerLaw` and the exposure fixes introduced there.
> Check with `python -m pytest src/gammabayes/tests -q`.

## 1. Execute the notebooks (once, on a machine with the package installed)

The HTML build **never** executes notebooks (`nbsphinx_execute = 'never'`), so run them first.
The full set takes a few minutes on a laptop CPU:

```bash
cd GammaBayes/docs
bash run_notebooks.sh                                   # all tutorials
bash run_notebooks.sh 06_Likelihood_Batching_Autograd.ipynb   # one notebook
```

`run_notebooks.sh` runs `jupyter nbconvert --to notebook --execute --inplace` on every
`tutorials/*.ipynb`, stops at the first failure, then calls `annotate_timings.py`, which appends
`[wall time: ...]` to each code cell's output and writes the per-notebook totals into
`tutorials/README.md`.  `GAMMABAYES_DOCS_DEVICE`
selects cpu / cuda / mps (Apple silicon runs in float32).  Notebook 04 needs the DM spectral tables
shipped with the package (CosmiXs / PPPC); notebook 02 needs the CTAO prod5 IRF tarball
(`package_data/CTAO_irf_fits_files/prod5`), which is unpacked on first use.

## 2. Build the HTML

```bash
pip install -r docs/requirements.txt      # sphinx, nbsphinx, myst-parser, rtd theme
module load pandoc  # or: conda install pandoc
cd docs && make html
open _build/html/index.html
```

`conf.py` imports `gammabayes` from `../src` for the API pages. Any heavy dependency that is not
installed in the documentation environment (torch, gammapy, ...) is mocked automatically, so a
lightweight environment can still build the text; the analysis environment gives the full
signatures and cross references.

## 3. Adding a tutorial

Drop the notebook in `tutorials/`, add it to the toctree in `index.rst` and to the list in
`tutorials/README.md`, execute it, rebuild.
