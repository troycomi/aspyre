import functools
import logging.config
import logging
import json
from copy import deepcopy
from types import SimpleNamespace


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class ConfigContext:
    def __init__(self, config, d):
        self._original_namespace = deepcopy(config.namespace)
        self.config = config
        for k, v in d.items():
            rsetattr(self.config, k, v)

    def __enter__(self):
        return self.config

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.config.namespace = self._original_namespace


class Config:
    def __init__(self, json_string):
        d = json.loads(json_string)
        if 'logging' in d:
            logging.config.dictConfig(d['logging'])
        else:
            logging.basicConfig(level=logging.INFO)

        # Now that logging is configured, reload the json, but now with an object hook
        # so we have cleaner access to keys by way of (recursive) attributes
        self.namespace = json.loads(json_string, object_hook=lambda d: SimpleNamespace(**d))

    def __getattr__(self, item):
        return getattr(self.namespace, item)

    def override(self, override_dict):
        return ConfigContext(self, override_dict)
