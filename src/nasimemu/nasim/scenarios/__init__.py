from .utils import INTERNET as INTERNET
from .scenario import Scenario as Scenario
from .loader import ScenarioLoader as ScenarioLoader
from .loader_v2 import ScenarioLoaderV2 as ScenarioLoaderV2
from .generator import ScenarioGenerator as ScenarioGenerator
import nasimemu.nasim.scenarios.benchmark as benchmark

from pathlib import Path
from typing import Any

def make_benchmark_scenario(scenario_name: str, seed: int | None = None) -> Scenario:
    """Generate or Load a benchmark Scenario.

    Parameters
    ----------
    scenario_name : str
        the name of the benchmark environment
    seed : int, optional
        random seed to use to generate environment (default=None)

    Returns
    -------
    Scenario
        a new scenario instance

    Raises
    ------
    NotImplementederror
        if scenario_name does no match any implemented benchmark scenarios.
    """
    if scenario_name in benchmark.AVAIL_GEN_BENCHMARKS:
        params = benchmark.AVAIL_GEN_BENCHMARKS[scenario_name]
        params['seed'] = seed
        return generate_scenario(**params)
    elif scenario_name in benchmark.AVAIL_STATIC_BENCHMARKS:
        scenario_def = benchmark.AVAIL_STATIC_BENCHMARKS[scenario_name]
        return load_scenario(scenario_def["file"], name=scenario_name)
    else:
        raise NotImplementedError(
            f"Benchmark scenario '{scenario_name}' not available."
            f"Available scenarios are: {benchmark.AVAIL_BENCHMARKS}"
        )


def generate_scenario(num_hosts: int, num_services: int, **params: Any) -> Scenario:
    """Generate Scenario from network parameters.

    Parameters
    ----------
    num_hosts : int
        number of hosts to include in network (minimum is 3)
    num_services : int
        number of services to use in environment (minimum is 1)
    params : dict, optional
        generator params (see :class:`ScenarioGenertor` for full list)

    Returns
    -------
    Scenario
        a new scenario object
    """
    generator = ScenarioGenerator()
    return generator.generate(num_hosts, num_services, **params)


def load_scenario(path: str, name: str | None = None) -> Scenario:
    """Load NASim Environment from a .yaml scenario file.

    Parameters
    ----------
    path : str
        path to the .yaml scenario file
    name : str, optional
        the scenarios name, if None name will be generated from path
        (default=None)

    Returns
    -------
    Scenario
        a new scenario object
    """
    loader: ScenarioLoader | ScenarioLoaderV2
    if '.v2' in Path(path).suffixes:
        loader = ScenarioLoaderV2()
    else:
        loader = ScenarioLoader()

    return loader.load(path, name=name)


def get_scenario_max(scenario_name: str) -> Any:
    if scenario_name in benchmark.AVAIL_GEN_BENCHMARKS:
        return benchmark.AVAIL_GEN_BENCHMARKS[scenario_name]["max_score"]
    elif scenario_name in benchmark.AVAIL_STATIC_BENCHMARKS:
        return benchmark.AVAIL_STATIC_BENCHMARKS[scenario_name]["max_score"]
    return None
