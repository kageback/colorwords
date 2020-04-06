def make(env_module, **kwargs):
    if env_module == 'wcs':
        from com_enviroments.wcs import WCS_Enviroment
        return WCS_Enviroment(**kwargs)
    if env_module == 'wcs_prior':
        from com_enviroments.wcs_prior import WCS_Prior_Enviroment
        return WCS_Prior_Enviroment(**kwargs)
    elif env_module == 'numbers':
        from com_enviroments.numbers import NumberEnvironment
        return NumberEnvironment(**kwargs)