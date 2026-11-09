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
