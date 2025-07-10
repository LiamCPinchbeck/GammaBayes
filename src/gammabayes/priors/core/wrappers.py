import inspect
from functools import wraps
from gammabayes import update_with_defaults



def _wrap_if_missing_keyword(logfunc, kwarg_string):

    # Get the signature of the function
    signature = inspect.signature(logfunc)
    # Check if the keyword argument is in the function's parameters
    if kwarg_string in signature.parameters:
        # If the keyword argument is already present, return the function as is
        return logfunc
    else:
        # If not, wrap the function to accept the keyword argument
        @wraps(logfunc)
        def wrapper(*args, **kwargs):
            # Remove the keyword argument from kwargs if present
            if kwarg_string in kwargs:
                kwd_parameters = kwargs[kwarg_string]

                kwargs.pop(kwarg_string, None)

                kwargs.update(**kwd_parameters)
            # Call the original function with the remaining arguments
            return logfunc(*args, **kwargs)
        return wrapper