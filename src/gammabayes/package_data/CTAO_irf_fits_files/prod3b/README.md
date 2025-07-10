# CTAO Instrument Response Functions - version prod3b-v2

**Please check the CTA webpage ([https://www.cta-observatory.org/science/cta-performance/](https://www.cta-observatory.org/science/cta-performance/)) for the most recent instrument response functions.**

**Figures and files in this repository are superseeded by newer versions and provided for archival reasons.**

The CTA Observatory (CTAO) will provide very wide energy range and excellent angular resolution and sensitivity in comparison to any existing gamma-ray detector. Energies down to 20 GeV will allow CTAO to study the most distant objects. Energies up to 300 TeV will push CTAO beyond the edge of the known electromagnetic spectrum, providing a completely new view of the sky. 
This data repository provides access to performance evaluation and instrument response functions (IRFs) for CTA.

- IRF version: prod3b-v2
- Publication date: April 2019
- Archived webpage with performance figures included: [CTAO Performance Description (file Website.md)](Website.md)
- Licence: his work is licensed under a [Creative Commons Attribution 4.0 International License](LICENSE).

## Citation and Acknowledgements:

In cases for which the CTA instrument response functions are used in
a research project, we ask to add the following acknowledgement in
any resulting publication:

“This research has made use of the CTA instrument response functions
provided by the CTA Consortium and Observatory, 
see https://www.cta-observatory.org/science/cta-performance/ (version prod3b-v2; https://doi.org/10.5281/zenodo.5163273) for
more details.”

## Description

### Monte Carlo Simulations:

The performance values are derived from detailed Monte Carlo (MC)
simulations of the CTA instrument based on the CORSIKA air shower
code (v6.9+, with the hadronic interaction models QGSjet-II-04 and
URQMD, [1]) and telescope simulation tool sim\_telarray [2]. A power-
law gamma-ray spectrum with photon index 2.62 was assumed in the
calculations, although none of the instrument response functions
(e.g. differential flux sensitivities, effective areas, angular or
energy resolutions) depends on the assumed spectral shape of the
gamma-ray source. Background cosmic-ray spectra of proton and
electron/positron particle types are modelled according to recent
measurements from cosmic-ray instruments.

Nominal telescope pointing is assumed, with all telescopes pointing
directions parallel to each other (performance estimation for other
pointing modes, e.g. divergent pointing will be provided in the
future). Performance estimations are available for two zenith angles
(20 deg and 40 deg), and for each zenith angle for two different
azimuth angles (corresponding to pointing towards the magnetic North
and South). There are significant performance differences found
between the two azimuthal pointing directions  (especially for the
Northern site) as the impact of the geomagnetic field is large
enough to influence notably the air shower development. For general
studies, the  use of the azimuth-averaged instrument response
functions is recommended.

### Instrument Response Functions (IRFs):

The analysis has been tuned to maximize the performance in terms of
flux sensitivity. The optimal analysis cuts depend on the duration
of the observation, therefore the IRFs are provided for 3 different
observation times, from 0.5 to 50 h. IRFs are provided as binned
histogram or FITS tables. It should be stressed, that the full
potential of CTA in terms of angular and energy resolution is not
revealed by these IRFS, due to the focus on the optimisation for
best flux sensitivity.

In general all histograms are binned with a 0.2-binning on the
logarithmic energy axis (5 bins per decade); some selected
histograms (e.g. effective areas or energy migration matrices) are
provided with a finer binning. Effective area and energy migration
matrix are available in a double version: one for the case in which
there is no a priori knowledge of the true direction of incoming
gamma rays (e.g. for the observation of diffuse sources), and
another for observations of point-like objects (including among the
analysis cuts one on the angle between the true and the
reconstructed gamma-ray direction).

IRFs are provided in ROOT format, as FITS tables, and for some on-
axis IRFs also as simple ASCII files. The FITS tables can be used
directly as input to science analysis tools.
The values of the IRFs are
identical for the different file format, with one exception: the
angular point-spread function is approximated by a Gaussian function
for the FITS tables, while the ROOT files contain the full
distribution.

File Naming (examples):

- CTA-Performance-prod3b-v2-North-20deg-average-50h.root: IRF for CTA
Northern site on La Palma, 20 deg zenith angle, azimuth-averaged
pointing, optimised for 50 hours of observation time
- CTA-Performance-prod3b-v2-South-20deg-average-50h.root: IRF for CTA
Southern site in Paranal, 20 deg zenith angle, azimuth-averaged
pointing, optimised for 50 hours of observation time
- CTA-Performance-prod3b-v2-South-40deg-S-30m.root: RF for CTA
Southern site, 40 deg zenith angle, South pointing, optimised for 30
minutes of observation time

List of files:

- fits/CTA-Performance-prod3b-v2-FITS.tar.gz - IRFs in FITS format (making
use of the HEASARC’s caldb indexing) - includes IRFs for 20 deg, 40
deg, and 60 deg zenith angle, average, north and south pointing
- root/CTA-Performance-prod3b-v2-20deg-ROOT.tar.gz - IRFs in ROOT format
for 20 deg zenith angle, azimuth-averaged, north and south pointing
- root/CTA-Performance-prod3b-v2-40deg-ROOT.tar.gz - IRFs in ROOT format
for 40 deg zenith angle, azimuth-averaged, north and south pointing
- root/CTA-Performance-prod3b-v2-60deg-ROOT.tar.gz - IRFs in ROOT format
for 60 deg zenith angle, azimuth-averaged, north and south pointing
- ascii/CTA-Performance-prod3b-v2-20deg-ASCII.tar.gz - (selected) IRFs in
ASCII format for 20 deg zenith angle, azimuth-averaged, north and
south pointing
- ascii/CTA-Performance-prod3b-v2-40deg-ASCII.tar.gz - (selected) IRFs in
ASCII format for 40 deg zenith angle, azimuth-averaged, north and
south pointing
- ascii/CTA-Performance-prod3b-v2-60deg-ASCII.tar.gz - (selected) IRFs in
ASCII format for 60 deg zenith angle, azimuth-averaged, north and
south pointing

### CTA Science Performance Requirements

**Performance requirements for CTA are currently under review. The attached requirements are preliminary and subject to change.**

The following documents summarise the science performance requirements for CTA. These requirements correspond to the baseline implementation of CTA. Values for the requirements on differential sensitivity, angular and energy resolution are provided in plain text files.

- [Requirements description (CTA-SPE-SCI-00000-0001_Issue_1_SystemLevelSciencePerformanceReqs.pdf)](requirements/CTA-SPE-SCI-00000-0001_Issue_1_SystemLevelSciencePerformanceReqs.pdf)
- [CTA-Performance-Requirements.tar.gz (ascii files)](requirements/CTA-Performance-Requirements.tar.gz)

## References

- [1] https://www.ikp.kit.edu/corsika/
- [2] Bernloehr, K. 2008, Astroparticle Physics, 30, 149

## Acknowledgements


We would like to thank the computing centres that provided resources for the generation of the Instrument Response Functions:

- CAMK, Nicolaus Copernicus Astronomical Center, Warsaw, Poland
- CETA-GRID, Resource Center CETA-CIEMAT, Trujillo, Spain
- CIEMAT-LCG2, CIEMAT, Madrid, Spain
- CYFRONET-LCG2, ACC CYFRONET AGH, Cracow, Poland
- DESY-ZN, Deutsches Elektronen-Synchrotron, Standort Zeuthen, Germany
- GRIF, Grille de Recherche d’Ile de France, Paris, France
- IN2P3-CC, Centre de Calcul de l’IN2P3, Villeurbanne, France
- IN2P3-CPPM, Centre de Physique des Particules de Marseille, Marseille, France
- IN2P3-LAPP, Laboratoire d Annecy de Physique des Particules, Annecy, France
- INFN-FRASCATI, INFN Frascati, Frascati, Italy
- INFN-T1, CNAF INFN, Bologna, Italy
- INFN-TORINO, INFN Torino, Torino, Italy
- MPIK, Heidelberg, Germany
- M3PEC, Mesocentre Aquitain, Bordeaux, France
- OBSPM, Observatoire de Paris Meudon, Paris, France
- PIC, port d’informacio cientifica, Bellaterra, Spain
- prague_cesnet_lcg2, CESNET, Prague, Czech Republic
- praguelcg2, FZU Prague, Prague, Czech Republic
- SE-SNIC-T2, The Swedish WLCG Tier 2 InitiativeStockholm, Sweden
