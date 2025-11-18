"""Configuration loader.

Reads configuration files in YAML format and returns a dictionary.  If
the PyYAML library is not available, it falls back to a simple
dictionary (empty).  Configuration files should reside in the
`configs/` directory at the project root.
"""

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.  Returns an empty dict if
        PyYAML is not available or if the file cannot be read.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        return {}
    if yaml is None:
        # PyYAML not installed
        return {}
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}