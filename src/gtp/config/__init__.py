"""Configuration management for GTP."""

from .defaults import DEFAULTS
from .config import Config, load_config, objectview

__all__ = ["DEFAULTS", "Config", "load_config", "objectview"]
