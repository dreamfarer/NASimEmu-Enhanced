from nasimemu.nasim.envs.environment import NASimEnv
from nasimemu.nasim.scenarios import load_scenario


__all__ = ['load']

def load(path: str,
         fully_obs: bool = False,
         flat_actions: bool = True,
         flat_obs: bool = True,
         name: str | None = None) -> NASimEnv:
    """Load NASim Environment from a .v2.yaml scenario file.

    Parameters
    ----------
    path : str
        path to the .v2.yaml scenario file
    fully_obs : bool, optional
        The observability mode of environment, if True then uses fully
        observable mode, otherwise partially observable (default=False)
    flat_actions : bool, optional
        if true then uses a flat action space, otherwise will use
        parameterised action space (default=True).
    flat_obs : bool, optional
        if true then uses a 1D observation space. If False
        will use a 2D observation space (default=True)
    name : str, optional
        the scenarios name, if None name will be generated from path
        (default=None)

    Returns
    -------
    NASimEnv
        a new environment object
    """
    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs}
    scenario = load_scenario(path, name=name)
    return NASimEnv(scenario, **env_kwargs)
