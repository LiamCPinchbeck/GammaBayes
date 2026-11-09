"""Reader for the QCD-uncertainty annihilation-spectrum tables (github.com/ajueid/qcd-dm.github.io).

Reads ``data/wUncertainty/AtProduction-Ga.dat`` (shipped as ``QCDUNC_Tables/AtProduction-Ga.dat`` or
``AtProduction-Ga-Combined-Up-QCDUNC.dat``).  The table is whitespace-separated (two or more spaces)
with columns ``'# DM'`` [GeV], ``'x'`` (= E / m_DM) and, per channel, ``'dN/dx [ch]'`` and the signed
offsets ``'+DHad [ch]'``, ``'-DHad [ch]'``, ``'+DScale [ch]'``, ``'-DScale [ch]'``, ``'+DcNS [ch]'``,
``'-DcNS [ch]'`` (hadronisation, shower scale and non-singular contributions; the ``+`` offsets are
added to the central value, the ``-`` offsets subtracted, following the upstream convention).

The reader exposes the ``PPPCReader`` interface (``mass_axis`` in TeV, ``log10x_axis``, ``output_shape``,
``__getitem__``), converts dN/dx to dN/dlog10(x) and provides ``'<ch>_up'`` / ``'<ch>_down'`` spectra
with the three uncertainty sources combined in quadrature (treated as independent; Amoroso et al.,
JCAP 05 (2019) 007, arXiv:1812.07424).
"""

import numpy as np, pandas as pd

_CH = ['ee', 'mumu', 'tautau', 'uu', 'dd', 'ss', 'cc', 'bb', 'tt', 'gaga', 'zz', 'ww', 'gg', 'hh']


class QCDUNCReader:
    darkSUSY_to_QCDUNC_converter = {
        "e+e-": "ee", "mu+mu-": "mumu", "tau+tau-": "tautau",
        "uu": "uu", "dd": "dd", "ss": "ss", "cc": "cc", "bb": "bb", "tt": "tt",
        "gammagamma": "gaga", "ZZ": "zz", "W+W-": "ww", "gg": "gg", "HH": "hh"}

    def __init__(self, file_name):
        df = pd.read_table(file_name, sep=r'\s\s+', engine='python')
        dm = np.asarray(df['# DM'], dtype=float)
        x = np.asarray(df['x'], dtype=float)
        ln10x = np.log(10.0) * x                              # dN/dx -> dN/dlog10x
        self.data_dict = {'mDM': dm / 1000.0,                 # GeV -> TeV
                          'Log[10,x]': np.log10(np.clip(x, 1e-300, None))}
        for ch in _CH:
            if 'dN/dx [{}]'.format(ch) not in df.columns:
                continue
            cen = ln10x * np.asarray(df['dN/dx [{}]'.format(ch)], dtype=float)
            self.data_dict[ch] = cen
            comps = {}
            for src in ('DHad', 'DScale', 'DcNS'):
                comps['+' + src] = ln10x * np.asarray(df['+{} [{}]'.format(src, ch)], dtype=float)
                comps['-' + src] = ln10x * np.asarray(df['-{} [{}]'.format(src, ch)], dtype=float)
                self.data_dict['{}_{}_up'.format(ch, src)] = comps['+' + src]
                self.data_dict['{}_{}_down'.format(ch, src)] = comps['-' + src]
            up = np.sqrt(comps['+DHad'] ** 2 + comps['+DScale'] ** 2 + comps['+DcNS'] ** 2)
            dn = np.sqrt(comps['-DHad'] ** 2 + comps['-DScale'] ** 2 + comps['-DcNS'] ** 2)
            self.data_dict[ch + '_up'] = np.clip(cen + up, 0.0, None)
            self.data_dict[ch + '_down'] = np.clip(cen - dn, 0.0, None)

        self.mass_axis = np.unique(self['mDM'])
        self.log10x_axis = np.unique(self['Log[10,x]'])
        self.num_mass, self.num_log10x = len(self.mass_axis), len(self.log10x_axis)
        self.output_shape = (self.num_mass, self.num_log10x)
        assert self.num_mass * self.num_log10x == len(dm), \
            "x grid differs between mass blocks -- inspect the table"

    def __getitem__(self, key):
        return self.data_dict[key]

    def keys(self):
        return self.data_dict.keys()