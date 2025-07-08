from astropy import units as u
from gammabayes.priors.core.wrappers import _wrap_if_missing_keyword
from gammabayes import update_with_defaults



class BaseSpatial_PriorComp:

    def __init__(self, logfunc, *args, **kwargs):

        self._input_logfunc = logfunc


    def __call__(self, *args, **kwargs):


        return self._input_logfunc(*args, **kwargs)
