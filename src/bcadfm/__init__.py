from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]

try:
    __version__ = version("bcadfm")
except PackageNotFoundError:  # local, editable install during development
    __version__ = "0.0.0"
