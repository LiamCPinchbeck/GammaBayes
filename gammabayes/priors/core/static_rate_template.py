from .discrete_logprior import DiscreteLogPrior
import torch

class StaticDistTemplate(DiscreteLogPrior):


    def __init__(self, coord_geom, rate_tensor_template=None, rate_tensor_logtemplate=None):


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
            coord_geom          = coord_geom, 
            rate_tensor_func    = self.func_template, 
            rate_tensor_logfunc = self.logfunc_template)



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