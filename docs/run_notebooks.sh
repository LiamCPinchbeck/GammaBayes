#!/usr/bin/env bash
# Execute every tutorial notebook in place (outputs are stored in the .ipynb files, which is
# what the Sphinx build renders), then stamp each code cell with its wall time
# (docs/annotate_timings.py) and record the per-notebook totals in tutorials/README.md.
#   Usage:  bash docs/run_notebooks.sh [notebook ...]
#   Env:    GAMMABAYES_DOCS_FAST=1                smaller grids / fewer samples
#           GAMMABAYES_DOCS_DEVICE=cpu|cuda|mps   force the torch device (default: cuda > mps > cpu)
set -euo pipefail
# uses the python of the ACTIVE environment (a bare `jupyter` may resolve to another install, e.g. Anaconda)
python -c "import gammabayes, nbconvert, ipykernel" || { echo "install the package and docs requirements into this environment first (pip install -e 'src/[docs]')"; exit 1; }
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/tutorials"
export GAMMABAYES_DOCS_FAST=${GAMMABAYES_DOCS_FAST:-0}
nbs=("$@")
[[ ${#nbs[@]} -gt 0 ]] || nbs=( [0-9][0-9]_*.ipynb )
for nb in "${nbs[@]}"; do
  echo "=== $nb"
  python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
done
python "$here/annotate_timings.py" "${nbs[@]}"
echo "all notebooks executed"
