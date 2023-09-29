import sys

def get_available_options_from_module(module):
    from importlib import import_module

    _module = import_module(module)
    return {k[:-1] if k.endswith('_') else k: getattr(_module, k) for k in getattr(_module, '__all__')}


# decorator function that adds function/class to __all__ dictionary
def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def get_available_subclasses(base_module_name, base_class_name, _package, _globals):
    from importlib import import_module
    __all__ = []
    _base_cls = import_module(base_module_name, package=_package)
    _base_cls = getattr(_base_cls, base_class_name)
    _globals[base_class_name] = _base_cls
    _module = import_module(base_module_name, package=_package)
    for _k, _attr in _module.__dict__.items():
        try:
            if issubclass(_attr, _base_cls) and _k[0] != '_' and _base_cls.__name__ != _k:
                _globals[_k] = getattr(_module, _k)
                __all__.append(_k)
        except Exception as e:
            continue

    return __all__