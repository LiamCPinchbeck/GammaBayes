from .discrete_logprior import DiscreteLogPrior
from .source_flux_prior import SourceFluxDiscreteLogPrior
import torch

class StaticDistTemplate(DiscreteLogPrior):


    def __init__(self, rate_tensor_template=None, rate_tensor_logtemplate=None, *args, **kwargs):


        self.rate_tensor_template = rate_tensor_template
        self.rate_tensor_logtemplate = rate_tensor_logtemplate
        
        if rate_tensor_template is None:
            self.rate_tensor_template = self.__rate_tensor_template
        else:
            self.rate_tensor_template = rate_tensor_template

        if rate_tensor_logtemplate is None:
            self.rate_tensor_logtemplate = self.__rate_tensor_logtemplate
        else:
            self.rate_tensor_logtemplate = rate_tensor_logtemplate


        super().__init__(
            rate_tensor_func    = self.func_template, 
            rate_tensor_logfunc   = self.logfunc_template,
            *args,
            **kwargs
            )



    @property
    def __rate_tensor_template(self):
        return self.rate_tensor_logtemplate.exp()

    def func_template(self, *args, **kwargs):
        return self.rate_tensor_template

    @property
    def __rate_tensor_logtemplate(self):
        return self.rate_tensor_template.log()
        
    def logfunc_template(self, *args, **kwargs):
        return self.rate_tensor_logtemplate



class StaticSourceDistTemplate(SourceFluxDiscreteLogPrior):
    def __init__(self, source_flux_tensor_template=None, source_flux_tensor_logtemplate=None, *args, **kwargs):

        self.source_flux_tensor_template = source_flux_tensor_template
        self.source_flux_tensor_logtemplate = source_flux_tensor_logtemplate
        
        if source_flux_tensor_template is None:
            self.rate_tensor_template = self.__source_flux_tensor_template
        else:
            self.rate_tensor_template = source_flux_tensor_template

        if source_flux_tensor_logtemplate is None:
            self.source_flux_tensor_logtemplate = self.__source_flux_tensor_logtemplate
        else:
            self.source_flux_tensor_logtemplate = source_flux_tensor_logtemplate


        super().__init__(log_flux_function=self.logfunc_template, *args, **kwargs)


    @property
    def __source_flux_tensor_template(self):
        return self.source_flux_tensor_logtemplate.exp()

    def func_template(self, *args, **kwargs):
        return self.source_flux_tensor_template

    @property
    def __source_flux_tensor_logtemplate(self):
        return self.source_flux_tensor_template.log()
        
    def logfunc_template(self, *args, **kwargs):
        return self.source_flux_tensor_logtemplate
