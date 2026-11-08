"""YAML configuration loading with inheritance support.

Loads hierarchical YAML configs where child configs can ``inherit`` from a
parent.  The global ``config`` dict is populated once at the leaf node and
shared across the entire SLAM pipeline.
"""

import re
from typing import Any

import yaml

config: dict[str, Any] = {}


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads exponent floats without a dot (``1e-8``) as floats.

    A subclass, so the resolver is registered once and other SafeLoader users in the process are not affected.
    """


# from https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_config(path: str = "config/base.yaml", is_parent: bool = False) -> dict[str, Any]:
    """Load a YAML config file, recursively merging any inherited parent.

    Args:
        path: Filesystem path to the YAML config file.
        is_parent: Internal flag — when ``True`` the result is returned
            without updating the global ``config`` dict.

    Returns:
        The merged configuration dictionary.
    """
    with open(path) as f:
        cfg: dict[str, Any] = yaml.load(f, Loader=_Loader)
    inherit: str | None = cfg.get("inherit")
    if inherit is not None:
        cfg_parent: dict[str, Any] = load_config(inherit, is_parent=True)
    else:
        cfg_parent = dict()
    cfg = merge_config(cfg_parent, cfg)
    if is_parent:
        return cfg

    # update the global config only once at the child node
    global config
    config.update(cfg)

    return config


def merge_config(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``dict2`` into ``dict1``, mutating ``dict1`` in place.

    Args:
        dict1: Base dictionary (modified in place).
        dict2: Override dictionary whose values take precedence.

    Returns:
        The merged ``dict1``.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            merge_config(dict1[k], v)
        else:
            dict1[k] = v
    return dict1
