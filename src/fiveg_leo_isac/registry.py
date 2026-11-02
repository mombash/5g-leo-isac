"""Canonical receiver registry, backed by the fixed method configuration."""
import copy
from .assets import configuration

_CONFIG=configuration()
METHODS=tuple(_CONFIG['methods'])
CONTROLS=tuple(_CONFIG['controls'])
ADAPTIVE_METHODS=tuple(name for group in ['methods','controls']
                       for name,settings in _CONFIG[group].items()
                       if settings['implementation']=='adaptive')


def validate_name(name):
    if name not in METHODS+CONTROLS:
        raise ValueError('Unknown receiver method: '+str(name))
    return name


def method_settings(name):
    validate_name(name)
    group='controls' if name in CONTROLS else 'methods'
    return copy.deepcopy(_CONFIG[group][name])
