def make(env_module, **kwargs):
    if env_module == 'wcs':
        from com_enviroments.wcs import WCS_Enviroment
        return WCS_Enviroment(**kwargs)
    elif env_module == 'numbers':
        from com_enviroments.numbers import NumberEnvironment
        return NumberEnvironment(**kwargs)