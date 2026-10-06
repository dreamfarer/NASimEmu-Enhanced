from .utils import INTERNET as INTERNET
from .scenario import Scenario as Scenario
from .loader_v2 import ScenarioLoaderV2 as ScenarioLoaderV2

from pathlib import Path

def load_scenario(path: str, name: str | None = None) -> Scenario:
    """Load NASim Environment from a .v2.yaml scenario file.

    Parameters
    ----------
    path : str
        path to the .v2.yaml scenario file
    name : str, optional
        the scenarios name, if None name will be generated from path
        (default=None)

    Returns
    -------
    Scenario
        a new scenario object
    """
    if '.v2' not in Path(path).suffixes:
        raise ValueError(f"Only V2 scenarios (.v2.yaml) are supported: {path}")

    return ScenarioLoaderV2().load(path, name=name)
