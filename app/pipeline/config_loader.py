import functools
import os

import yaml

from app.config import settings

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")


def _resolve_config_path() -> str:
    config_path = settings.config_yaml_path or DEFAULT_CONFIG_PATH
    if not os.path.isabs(config_path):
        candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), config_path)
        if os.path.exists(candidate):
            return candidate
    return config_path


@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    config_path = _resolve_config_path()
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    return loaded or {}


def reload_config() -> dict:
    load_config.cache_clear()
    return load_config()