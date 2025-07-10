import shutil, os, sys
from ...__init__ import __version__


def CLI_Intro():
    print(
f"""
--------------------------------------------------------------------------------------------------------


Hey there! Thanks for installing GammaBayes version {__version__}. 

This package is primarily made to be custom on the level of the python interface.
However, we also try to offer some higher level interfaces from the level of configuration dictionaries in custom python scripts or
from configuration files that can be run from the command line. 

The latter is still under testing, but we believe it will be useful to have an example configuration file whether you wish
to use the CLI or to know what needs to be specified within the python scripts. So, an example named
"default_run_config.yaml" has been copied into your working directory. 

This config file is made to also work with slurm computing cluster in particular but if you just want to run the main scripts of
the GammaBayes CLI with specific config files without worrying about a slurm based computing cluster then the commands are

To run simulations
- $ gammabayes.run_simulate **path/to/config/file**

To marginalise over the nuisance parameters of simulate events
- $ gammabayes.run_marg **path/to/config/file**

To combine the results of the nuisance parameter marginalisation and perform nested sampling over specified parameters.
- $ gammabayes.run_combine **path/to/config/file**

We've had to make some relatively specific decisions in relation to these scripts however. So if you do wish
to interact with GammaBayes via the CLI we highly recommend going to the documentation.



Relatively in-depth documentation is available at https://gammabayes.readthedocs.io/  .

If you have any issues please raise them in the GitHub repo at https://github.com/lpin0002/GammaBayes/issues  .

Hope you're having a good day :)

--------------------------------------------------------------------------------------------------------

""")
    if not os.path.isfile(os.getcwd()+"/default_run_config.yaml"):
        shutil.copyfile(os.path.dirname(__file__)+"/default_run_config.yaml", os.getcwd()+"/default_run_config.yaml")
    else:
        print("""
default_run_config.yaml already in working directory. If you wish to get an example file then please remove this file.
              
""")
    