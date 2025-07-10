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

