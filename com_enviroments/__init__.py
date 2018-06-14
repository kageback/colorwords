import importlib
import com_enviroments
from com_enviroments.wcs import WCS_Enviroment

def make(env_module, **kwargs):
    if env_module == 'wcs':
        return WCS_Enviroment(**kwargs)

        #return importlib.import_module('com_enviroments' + '.' + env_module)
