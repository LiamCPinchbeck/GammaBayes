"""
qcdunc_reader.py — reader for QCDUNC wUncertainty tables (github.com/ajueid/qcd-dm.github.io).
File: data/wUncertainty/AtProduction-Ga.dat  ->  <data_path>/QCDUNC_Tables/AtProduction-Ga.dat
Goes beside PPPC_reader.py.

Column-NAME parsing (matches their Interpolate.py): whitespace table (sep=2+ spaces),
columns '# DM' [GeV], 'x' (=E/mDM), and per channel:
  'dN/dx [ch]', '+DHad [ch]','-DHad [ch]', '+DScale [ch]','-DScale [ch]', '+DcNS [ch]','-DcNS [ch]'
+D* are signed offsets to ADD, -D* signed offsets to SUBTRACT (their convention).
Exposes PPPCReader-like interface (mass_axis [TeV], log10x_axis, output_shape, __getitem__),
spectra converted dN/dx -> dN/dlog10x (x*ln10). Synthesizes '<ch>_up'/'<ch>_down' with the
three sources (hadronization, shower scale, non-singular) combined in quadrature
(independent; Amoroso et al. JCAP 05(2019)007, arXiv:1812.07424).
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