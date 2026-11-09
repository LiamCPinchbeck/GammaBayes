"""Reader for the CosmiXs annihilation-spectrum tables.

``CosmiXsReader`` exposes the same interface as ``PPPCReader`` (``mass_axis`` in TeV, ``log10x_axis``,
``output_shape``, ``__getitem__``), so the two tables are interchangeable in ``SingleDMSpectralComp``.

The CosmiXs tables (github.com/ajueid/CosmiXs; Arina et al., JCAP 03 (2024) 035, arXiv:2312.01153) are
whitespace-separated with columns: DM mass [GeV], x = E_kin / m_DM (100 logarithmic bins), then one
column per channel (29 in total, including the polarised eL/eR, WL/WT and ZL/ZT states) holding
dN/dlog10(x), the same units as PPPC4DMID.  The reader converts the mass column to TeV and names the
second column ``'Log[10,x]'`` so that the downstream interpolation (square-root interpolation on
(log10 mass, log10 x), positivity by squaring, dN/dlog10(x) -> dN/dE) is identical for both tables.
"""

import numpy as np
from .PPPC_reader import PPPCReader


class CosmiXsReader(PPPCReader):
    # darkSUSY channel name -> CosmiXs column (unpolarised / summed columns, matching the PPPC convention).
    # The available column names of a table are listed by ``CosmiXsReader(path).keys()``.
    darkSUSY_to_CosmiXs_converter = {
        "e+e-": "dNdLog10x[e]", "mu+mu-": "dNdLog10x[mu]", "tau+tau-": "dNdLog10x[tau]",
        "nuenue": "dNdLog10x[nue]", "numunumu": "dNdLog10x[numu]", "nutaunutau": "dNdLog10x[nutau]",
        "uu": "dNdLog10x[u]", "dd": "dNdLog10x[d]", "ss": "dNdLog10x[s]",
        "cc": "dNdLog10x[c]", "bb": "dNdLog10x[b]", "tt": "dNdLog10x[t]",
        "gg": "dNdLog10x[g]", "gammagamma": "dNdLog10x[a]",
        "W+W-": "dNdLog10x[W]", "ZZ": "dNdLog10x[Z]", "HH": "dNdLog10x[H]",
    }
    def read_data(self):
        data_dict = super().read_data()   # whitespace table; divides mDM by 1000 ONLY if header=='mDM'
        keys = list(data_dict.keys())
        # normalize column names/units to PPPC conventions
        if 'mDM' not in data_dict:        # column 1 = mass [GeV] by construction
            data_dict['mDM'] = np.asarray(data_dict.pop(keys[0]), dtype=float) / 1000.0  # GeV -> TeV
        if 'Log[10,x]' not in data_dict:  # column 2 = x (or log10 x) by construction
            col = np.asarray(data_dict.pop(keys[1]), dtype=float)
            data_dict['Log[10,x]'] = np.log10(col) if col.max() > 0 else col
        return data_dict