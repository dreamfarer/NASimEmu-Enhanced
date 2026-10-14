# This is a high-level wrapper around SimulationEnvironment that forwards reset() and step() to the EmulationNetwork instead of the simulator.

from typing import Any
from nasimemu_enhanced.simulation.environment import SimulationEnvironment

import logging, time
from numpy.typing import NDArray

from nasimemu_enhanced.emulation.network import EmulationNetwork

class EmulationEnvironment(SimulationEnvironment):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        assert self.current_state is not None
        self.emulated_network = EmulationNetwork(self.current_state.tensor.shape[1], self.scenario)
        self.logger = logging.getLogger("EmulationEnvironment")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[NDArray[Any], dict[str, Any]]:
        super().reset(seed=seed)

        self.logger.info("reset()")
        s = self.emulated_network.initial_scan()
        return s, {}

    def step(self, a: Any) -> tuple[NDArray[Any], float, bool, bool, dict[str, Any]]:
        time.sleep(2)

        self.logger.info(f"step() with {a}")
        s, i = self.emulated_network.translate_action(a)

        return s, 0., False, False, i  # s, r, terminated, truncated, i
