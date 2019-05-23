__version__ = "0.3.0"

import logging.config
import logging
from types import SimpleNamespace
import json
from importlib_resources import read_text

import aspyre


def setup_config():
    s = read_text(aspyre, 'config.json')
    d = json.loads(s)
    if 'logging' in d:
        logging.config.dictConfig(d['logging'])
    else:
        logging.basicConfig(level=logging.INFO)

    # Now that logging is configured, reload the json, but now with an object hook
    # so we have cleaner access to keys by way of (recursive) attributes
    config = json.loads(s, object_hook=lambda d: SimpleNamespace(**d))

    return config


config = setup_config()
