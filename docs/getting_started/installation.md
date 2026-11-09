# Installation

## Requirements

GammaBayes 2.1 targets **Python 3.10 - 3.13** and

| package | version | used for |
|---|---|---|
| `torch` | >= 2.4 | every model evaluation (CPU or CUDA) |
| `numpy` | >= 2.0 | table readers, IRF glue |
| `astropy` | >= 6.1 | units, FITS |
| `gammapy` | >= 2.0 (needs `regions < 0.11`) | reading / evaluating the CTAO IRF files |
| `h5py`, `healpy`, `scipy`, `pandas` | >= 3.11, >= 1.17, >= 1.11, >= 2.0 | templates and table readers |
| `pyro-ppl`, `dynesty`, `corner`, `matplotlib`, `seaborn`, `tqdm`, `icecream`, `requests` | any | sampling helpers, plotting, downloads |

Version pins were replaced by lower bounds in 2.1.0.

## Install

```bash
git clone https://github.com/lpin0002/GammaBayes.git
cd GammaBayes
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick the CUDA build for your driver
pip install -e src/                                                      # editable install of gammabayes
pip install -e 'src/[docs,test]'                                         # + sphinx/nbconvert and pytest (optional)
```

A plain `pip install -e src/` on a machine without a GPU installs the CPU wheel of torch, which is
sufficient for the tutorials (a few minutes each).

```{note}
**Older GPUs (e.g. Tesla P100, sm_60).**  Recent CUDA-12.8 torch wheels no longer ship Pascal
kernels; use a `cu126` build (`torch 2.14.x+cu126`) in a separate environment on those nodes.
```

```{note}
**Apple silicon (MPS).**  The torch core runs on the Metal backend: set
`torch.set_default_device("mps")` *and* `torch.set_default_dtype(torch.float32)` before building
any object (MPS has no float64), and export `PYTORCH_ENABLE_MPS_FALLBACK=1` so the few operators
without a Metal kernel fall back to the CPU.  Tutorials 01-05 do this automatically
(`GAMMABAYES_DOCS_DEVICE=mps`); tutorial 06 (profile likelihoods and autograd Hessians) needs
float64 and falls back to the CPU.  Expect float32 accuracy (see the precision notes in
[Migrating from 0.x](migration.md)); use the CPU for float64 reference runs.
```

```{note}
**pandas 3** removed `errors='ignore'` from `pd.to_numeric`; GammaBayes >= 2.1.0 handles this.  If you see `ValueError: invalid error value specified` while
loading the DM spectral tables, update.
```

## Data files

* **CTAO IRFs** (prod5 / prod3b) ship with the package under
  `gammabayes/package_data/CTAO_irf_fits_files/` as tarballs and are unpacked on first use. 
  If used in a published work or similar, please reference the original Zotero databases and acknowledge usage accordingly.
    * Zenodo Listing: https://zenodo.org/records/5499840 (DOI - https://doi.org/10.5281/zenodo.5499839)
* **DM spectra**: PPPC4DMID (`dark_matter/PPPC_Tables/`), CosmiXs
  (`dark_matter/CosmiXs_Tables/AtProduction-Gamma_COSMIXs.dat`) and the QCD-uncertainty band
  (`dark_matter/QCDUNC_Tables/`).  If a table is missing, the corresponding parameter (`kappa` for the
  code choice, `delta` for the QCD band) has **no effect** and a warning is printed once.  
  If used in a published work or similar, please reference the relevant papers, some BibTeX should be in the ReadME but you can also look at the arXiv listings.
     * CosmiXs: https://arxiv.org/abs/2312.01153
     * QCDUNC: https://arxiv.org/abs/2202.11546, https://arxiv.org/abs/2303.11363
     * PPPC4DMID: https://arxiv.org/abs/1012.4515, https://arxiv.org/abs/1009.0224
* **GALPROP** diffuse templates are downloaded on demand (~130 MB for "Medium"); on a cluster
  node without internet access, run one GALPROP-based model on the login node first so the file is
  cached. If used in a published work or similar, please reference the relevant papers, some BibTeX should be in the ReadME but you can also look at the arXiv listings (v57 and webrun).
    * GALPROP 57: https://arxiv.org/abs/2112.12745
    * GALPROP webrun: https://arxiv.org/abs/1008.3642


## Check the installation

```bash
pytest src/gammabayes/tests -q          # ~1 min on CPU; no data files needed
python -c "import gammabayes, torch; print(gammabayes.__version__, torch.cuda.is_available())"
```
