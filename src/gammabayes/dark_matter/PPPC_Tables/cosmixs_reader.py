"""
CosmiXsReader — drop-in replacement for PPPCReader, exposing the SAME interface
(mass_axis [TeV], log10x_axis, output_shape, __getitem__) so SingleDMSpectralComp
needs only a 2-line change.

CosmiXs tables (github.com/ajueid/CosmiXs; Arina et al., JCAP 03(2024)035,
arXiv:2312.01153): AtProduction-Gamma_COSMIXs.dat (renamed from AtProduction-Gamma.dat), whitespace table, columns =
DM mass [GeV], x = E_kin/mDM (100 log bins), then 29 channels (incl. polarized
eL/eR, WL/WT, ZL/ZT), spectra in dN/dlog10(x) — same units as PPPC.

WHERE IT GOES — in gammabayes .../spectral_components/<single_channel file>.py
(the file defining SingleDMSpectralComp):

    # 1. next to `from .PPPC_reader import PPPCReader` add:
    from .cosmixs_reader import CosmiXsReader

    # 2. replace the line
    #    atprod_gammas = PPPCReader(single_channel_spectral_data_path+"/PPPC_Tables/AtProduction_gamma_EW_corrections.dat")
    # with:
    atprod_gammas = CosmiXsReader(single_channel_spectral_data_path + "/CosmiXs_Tables/AtProduction-Gamma_COSMIXs.dat")

    # 3. replace `PPPCReader.darkSUSY_to_PPPC_converter[self.channel]`
    # with:
    CosmiXsReader.darkSUSY_to_CosmiXs_converter[self.channel]

Everything downstream (sqrt-interpolation on (log10 mass, log10 x), positivity
squaring, dN/dlog10x -> dN/dE conversion) is unchanged: this reader normalizes
the table to PPPC conventions (mDM -> TeV; second column -> 'Log[10,x]').
"""
import numpy as np
from .PPPC_reader import PPPCReader   # adjust to plain import if used standalone


class CosmiXsReader(PPPCReader):
    # darkSUSY channel name -> CosmiXs column. Unpolarized/summed columns chosen to
    # match PPPC's convention. VERIFY the exact header strings once with
    # `print(CosmiXsReader(path).keys())` and correct any that differ.
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