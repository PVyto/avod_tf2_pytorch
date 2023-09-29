__all__ = []


def _load_all_submodules():
    from pathlib import Path
    from importlib import import_module

    g = globals()
    package_path = Path(__file__).resolve().parent
    for pyfile in package_path.glob('*.py'):
        module_name = pyfile.stem
        if module_name == '__init__':
            continue
        module = import_module(f'.{module_name}', __package__)
        names = getattr(
            module, '__all__',
            (n for n in dir(module) if n[:1] != '_'))
        for name in names:
            g[name] = getattr(module, name)
            __all__.append(name)


_load_all_submodules()
del _load_all_submodules