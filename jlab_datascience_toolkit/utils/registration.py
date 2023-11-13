import importlib
import logging

module_log = logging.getLogger("Module Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class ModuleSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of data module with appropriate kwargs"""
        if self.entry_point is None:
            raise data_log.error('Attempting to make deprecated module {}. \
                               (HINT: is there a newer registered version \
                               of this module?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class ModuleRegistry(object):
    def __init__(self):
        self.module_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            module_log.info('Making new module: %s (%s)', path, kwargs)
        else:
            module_log.info('Making new module: %s', path)
        module_spec = self.spec(path)
        module = module_spec.make(**kwargs)

        return module

    def all(self):
        return self.module_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise module_log.error('A module ({}) was specified for the module but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `module.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.module_specs[id]
        except KeyError:
            raise module_log.error('No registered module with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.module_specs:
            raise module_log.error('Cannot re-register id: {}'.format(id))
        self.module_specs[id] = ModuleSpec(id, **kwargs)


# Global  registry
module_registry = ModuleRegistry()


def register(id, **kwargs):
    return module_registry.register(id, **kwargs)


def make(id, **kwargs):
    return module_registry.make(id, **kwargs)


def spec(id):
    return module_registry.spec(id)

def list_registered_modules():
    return list(module_registry.module_specs.keys())
