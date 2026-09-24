from .environment import NASimEnv
from nasimemu.nasim.scenarios import make_benchmark_scenario
from nasimemu.nasim.scenarios.scenario import Scenario


class NASimGymEnv(NASimEnv):
    """A wrapper around the NASimEnv compatible with OpenAI gym.make()

    See nasim.NASimEnv for details.
    """

    def __init__(self,
                 scenario: Scenario | str,
                 fully_obs: bool = False,
                 flat_actions: bool = True,
                 flat_obs: bool = True) -> None:
        """
        Parameters
        ----------
        scenario : str or or nasim.scenarios.Scenario
            either the name of benchmark environment (str) or a nasim Scenario
            instance
        fully_obs : bool, optional
            the observability mode of environment, if True then uses fully
            observable mode, otherwise partially observable (default=False)
        flat_actions : bool, optional
            if true then uses a flat action space, otherwise will use
            parameterised action space (default=True).
        flat_obs : bool, optional
            if true then uses a 1D observation space. If False
            will use a 2D observation space (default=True)
        """
        if not isinstance(scenario, Scenario):
            scenario = make_benchmark_scenario(scenario)
        super().__init__(scenario,
                         fully_obs=fully_obs,
                         flat_actions=flat_actions,
                         flat_obs=flat_obs)
