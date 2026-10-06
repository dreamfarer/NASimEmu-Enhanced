"""This module contains functions and classes for rendering NASim """
from typing import TYPE_CHECKING, Any

from prettytable import PrettyTable

if TYPE_CHECKING:
    from .network import Network
    from .observation import Observation
    from .state import State


class Viewer:
    """A class for visualizing the network state from NASimEnv"""

    def __init__(self, network: "Network") -> None:
        """
        Arguments
        ---------
        network : Network
            network of environment
        """
        self.network = network

    def render_readable(self, obs: "Observation") -> None:
        """Print a readable tabular version of observation to stdout

        Arguments
        ---------
        obs : Observation
            observation to view
        """
        host_obs, aux_obs = obs.get_readable()
        aux_table = self._construct_table_from_dict(aux_obs)
        host_table = self._construct_table_from_list_of_dicts(host_obs)
        print("Observation:")
        print(aux_table)
        print(host_table)

    def render_readable_state(self, state: "State") -> None:
        """Print a readable tabular version of observation to stdout

        Arguments
        ---------
        state : State
            state to view
        """
        host_obs = state.get_readable()
        host_table = self._construct_table_from_list_of_dicts(host_obs)
        print("State:")
        print(host_table)

    def _construct_table_from_dict(self, d: dict[str, Any]) -> PrettyTable:
        headers = list(d.keys())
        table = PrettyTable(headers)
        row = [str(d[k]) for k in headers]
        table.add_row(row)
        return table

    def _construct_table_from_list_of_dicts(
        self, l: list[dict[str, Any]]
    ) -> PrettyTable:
        headers = list(l[0].keys())
        table = PrettyTable(headers)
        for d in l:
            row = [str(d[k]) for k in headers]
            table.add_row(row)
        return table
